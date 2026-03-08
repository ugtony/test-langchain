import asyncio
import json
import uuid
from typing import TypedDict, List, Annotated

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.types import StreamWriter

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. 狀態定義 (保持支援 messages)
# ==========================================
class SubState(TypedDict):
    terms: List[str]

class MainState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    terms: List[str]

# ==========================================
# 2. 建立子圖與主圖 (直接注入 StreamWriter)
# ==========================================
async def research_node(state: SubState, writer: StreamWriter):
    # 寫法極度乾淨，不需要再傳遞 config
    writer({"node": "research_node", "msg": "正在掃描機台感測器與歷史 Recipe 紀錄..."})
    await asyncio.sleep(1.0) 
    return {"terms": ["Gas Flow", "RF Power", "Pressure"]}

sub_builder = StateGraph(SubState)
sub_builder.add_node("research", research_node)
sub_builder.add_edge(START, "research")
sub_builder.add_edge("research", END)
sub_graph = sub_builder.compile()

async def mock_llm_node(state: MainState, writer: StreamWriter):
    writer({"node": "llm_node", "msg": "正在綜合評估蝕刻深度與參數關聯..."})
    await asyncio.sleep(1.0)
    
    terms = state.get("terms", [])
    mock_reply = f"分析完成。建議針對 {', '.join(terms)} 進行 Recipe 微調，以確保良率。"
    return {"messages": [AIMessage(content=mock_reply)]}

workflow = StateGraph(MainState)
workflow.add_node("clarify", sub_graph) 
workflow.add_node("llm", mock_llm_node)

workflow.add_edge(START, "clarify")
workflow.add_edge("clarify", "llm")
workflow.add_edge("llm", END)

memory = MemorySaver()
graph_app = workflow.compile(checkpointer=memory)

# ==========================================
# 3. 安全 JSON 編碼器 (這個必須留著防崩潰)
# ==========================================
def safe_json_encoder(obj):
    if isinstance(obj, BaseMessage):
        return {"type": obj.type, "content": obj.content}
    try:
        return str(obj)
    except Exception:
        return "[Unserializable]"

# ==========================================
# 4. 極簡版 SSE API (拔除 Queue，直接 yield)
# ==========================================
@app.post("/chat")
async def chat(data: dict):
    user_input = data.get("input", "")
    thread_id = data.get("thread_id", str(uuid.uuid4()))
    config = {"configurable": {"thread_id": thread_id}}

    async def event_generator():
        try:
            # 🌟 簡化核心：直接在迴圈中 astream，交給 Python 3.12 原生處理非同步上下文！
            async for namespace, mode, chunk in graph_app.astream(
                {"messages": [HumanMessage(content=user_input)]},
                config,
                stream_mode=["updates", "custom", "messages"],
                subgraphs=True
            ):
                if mode == "messages":
                    msg_chunk, _ = chunk
                    chunk_data = {"type": msg_chunk.type, "content": msg_chunk.content}
                else:
                    chunk_data = chunk

                payload = {
                    "namespace": namespace,
                    "mode": mode,
                    "chunk": chunk_data
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False, default=safe_json_encoder)}\n\n"
                
        except Exception as e:
            yield f"data: {json.dumps({'mode': 'error', 'msg': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# ==========================================
# 5. 測試前端 (保持不變)
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def get_test_page():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LangGraph SSE 串流測試</title>
        <style>
            body { font-family: sans-serif; max-width: 800px; margin: 40px auto; background: #f4f4f9;}
            .box { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
            .log { height: 150px; overflow-y: auto; background: #1e1e1e; color: #00ff00; padding: 10px; font-family: monospace; font-size: 13px; margin-top: 5px; }
            button { padding: 10px 20px; cursor: pointer; background: #007bff; color: white; border: none; border-radius: 4px; }
            input { width: 70%; padding: 10px; margin-right: 10px; border: 1px solid #ccc; border-radius: 4px; }
            .error-text { color: #ff4444; font-weight: bold; }
        </style>
    </head>
    <body>
        <h2>LangGraph 串流測試面板 (三模式並存)</h2>
        <div class="box">
            <input type="text" id="userInput" value="請協助分析最新蝕刻製程的 Recipe 參數" />
            <button onclick="startStream()">發送請求</button>
        </div>
        <div class="box"><h3>Custom 模式 (進度事件)</h3><div id="customLog" class="log" style="color: #ffaa00;"></div></div>
        <div class="box"><h3>Updates 模式 (狀態更新)</h3><div id="updatesLog" class="log" style="color: #55b2ff;"></div></div>
        <div class="box"><h3>Messages 模式 (對話內容)</h3><div id="messagesLog" class="log"></div></div>

        <script>
            async function startStream() {
                const input = document.getElementById('userInput').value;
                document.getElementById('customLog').innerHTML = '';
                document.getElementById('updatesLog').innerHTML = '';
                document.getElementById('messagesLog').innerHTML = '';

                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ input: input })
                    });

                    const reader = response.body.getReader();
                    const decoder = new TextDecoder("utf-8");
                    let buffer = "";

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        
                        buffer += decoder.decode(value, { stream: true });
                        const chunks = buffer.split("\\n\\n");
                        buffer = chunks.pop();

                        for (const chunk of chunks) {
                            if (chunk.startsWith("data: ")) {
                                const data = JSON.parse(chunk.substring(6));
                                if (data.mode === 'error') {
                                    const errorMsg = `<span class="error-text">[系統錯誤] ${data.msg}</span><br>`;
                                    document.getElementById('customLog').innerHTML += errorMsg;
                                    document.getElementById('updatesLog').innerHTML += errorMsg;
                                    document.getElementById('messagesLog').innerHTML += errorMsg;
                                    continue;
                                }
                                const logLine = `[${data.namespace.join('/') || 'root'}] ${JSON.stringify(data.chunk)}<br>`;
                                if (data.mode === 'custom') document.getElementById('customLog').innerHTML += logLine;
                                else if (data.mode === 'updates') document.getElementById('updatesLog').innerHTML += logLine;
                                else if (data.mode === 'messages') document.getElementById('messagesLog').innerHTML += logLine;
                            }
                        }
                    }
                } catch (error) {
                    console.error("連線錯誤:", error);
                }
            }
        </script>
    </body>
    </html>
    """
    return html_content

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)