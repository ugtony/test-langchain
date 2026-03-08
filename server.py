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
from langgraph.types import StreamWriter, interrupt, Command
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
# 1. 狀態定義
# ==========================================
class SubState(TypedDict):
    terms: List[str]

class MainState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    terms: List[str]

# ==========================================
# 2. 建立子圖 (加入人類審核節點)
# ==========================================
async def research_node(state: SubState, writer: StreamWriter):
    writer({"node": "research_node", "msg": "正在掃描機台感測器與歷史 Recipe 紀錄..."})
    await asyncio.sleep(1.0) 
    return {"terms": ["Gas Flow", "RF Power", "Pressure"]}

async def human_review_node(state: SubState, writer: StreamWriter):
    writer({"node": "human_review_node", "msg": f"抓取到預設參數：{state['terms']}，準備暫停等待確認..."})
    
    # 觸發中斷！執行緒在此凍結
    human_feedback = interrupt("請確認是否同意使用這些參數，或提供修改後的參數陣列")
    
    # 收到 resume 指令後從這裡甦醒
    writer({"node": "human_review_node", "msg": f"收到工程師回覆，最終採用參數：{human_feedback}"})
    return {"terms": human_feedback}

sub_builder = StateGraph(SubState)
sub_builder.add_node("research", research_node)
sub_builder.add_node("review", human_review_node)
sub_builder.add_edge(START, "research")
sub_builder.add_edge("research", "review")
sub_builder.add_edge("review", END)
sub_graph = sub_builder.compile()

# ==========================================
# 3. 建立主圖
# ==========================================
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

def safe_json_encoder(obj):
    if isinstance(obj, BaseMessage):
        return {"type": obj.type, "content": obj.content}
    try:
        return str(obj)
    except Exception:
        return "[Unserializable]"

# ==========================================
# 4. API 路由 (支援 resume 指令)
# ==========================================
@app.post("/chat")
async def chat(data: dict):
    user_input = data.get("input", "")
    resume_data = data.get("resume")
    thread_id = data.get("thread_id", str(uuid.uuid4()))
    config = {"configurable": {"thread_id": thread_id}}

    async def event_generator():
        try:
            # 判斷是全新對話還是接續中斷
            if resume_data is not None:
                payload = Command(resume=resume_data)
            else:
                payload = {"messages": [HumanMessage(content=user_input)]}

            async for namespace, mode, chunk in graph_app.astream(
                payload,
                config,
                stream_mode=["updates", "custom", "messages"],
                subgraphs=True
            ):
                if mode == "messages":
                    msg_chunk, _ = chunk
                    chunk_data = {"type": msg_chunk.type, "content": msg_chunk.content}
                else:
                    chunk_data = chunk

                payload_data = {
                    "namespace": namespace,
                    "mode": mode,
                    "chunk": chunk_data
                }
                yield f"data: {json.dumps(payload_data, ensure_ascii=False, default=safe_json_encoder)}\n\n"
                
        except Exception as e:
            yield f"data: {json.dumps({'mode': 'error', 'msg': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# ==========================================
# 5. 內建瀏覽器測試介面 (加入中斷與恢復 UI)
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def get_test_page():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LangGraph Human-in-the-loop 測試</title>
        <style>
            body { font-family: sans-serif; max-width: 800px; margin: 40px auto; background: #f4f4f9;}
            .box { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
            .log { height: 150px; overflow-y: auto; background: #1e1e1e; color: #00ff00; padding: 10px; font-family: monospace; font-size: 13px; margin-top: 5px; }
            button { padding: 10px 20px; cursor: pointer; border: none; border-radius: 4px; font-weight: bold; }
            .btn-primary { background: #007bff; color: white; }
            .btn-success { background: #28a745; color: white; }
            input[type="text"] { width: 70%; padding: 10px; margin-right: 10px; border: 1px solid #ccc; border-radius: 4px; }
            .error-text { color: #ff4444; font-weight: bold; }
            
            /* 中斷審核區塊的樣式 */
            #interruptArea { display: none; background: #fff3cd; border: 2px solid #ffe69c; }
            .interrupt-title { color: #856404; margin-top: 0; display: flex; align-items: center; }
        </style>
    </head>
    <body>
        <h2>蝕刻製程參數分析 (支援工程師審核)</h2>
        
        <div class="box" id="actionArea">
            <input type="text" id="userInput" value="請協助分析最新蝕刻製程的 Recipe 參數" />
            <button class="btn-primary" onclick="startStream()">發送請求</button>
        </div>
        
        <div class="box" id="interruptArea">
            <h3 class="interrupt-title">⚠️ 系統已暫停：等待工程師審核參數</h3>
            <p style="color: #666; font-size: 14px;">請修改下方陣列 (JSON 格式)，並點擊繼續執行：</p>
            <input type="text" id="resumeInput" value='["Temperature", "RF Power", "Pressure"]' />
            <button class="btn-success" onclick="resumeStream()">確認並繼續執行</button>
        </div>
        
        <div class="box"><h3>Custom 模式 (進度事件)</h3><div id="customLog" class="log" style="color: #ffaa00;"></div></div>
        <div class="box"><h3>Updates 模式 (狀態更新)</h3><div id="updatesLog" class="log" style="color: #55b2ff;"></div></div>
        <div class="box"><h3>Messages 模式 (對話內容)</h3><div id="messagesLog" class="log"></div></div>

        <script>
            // 🌟 關鍵：前端必須在全域記住當前的 thread_id
            let currentThreadId = "";

            // 處理 SSE 串流的核心函數
            async function processStream(payload) {
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
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
                                    return;
                                }

                                const nsStr = data.namespace.join('/') || 'root';
                                
                                // 🌟 偵測中斷訊號
                                if (data.mode === 'updates' && data.chunk.__interrupt__) {
                                    const logLine = `<span style="color: #ff4444; font-weight: bold;">[${nsStr}] ✋ 觸發中斷：等待人類指令...</span><br>`;
                                    document.getElementById('updatesLog').innerHTML += logLine;
                                    
                                    // 顯示審核控制台，並隱藏發送按鈕
                                    document.getElementById('interruptArea').style.display = 'block';
                                    document.getElementById('actionArea').style.display = 'none';
                                    return; // 結束讀取，因為後端已經凍結了
                                }

                                const logLine = `[${nsStr}] ${JSON.stringify(data.chunk)}<br>`;
                                
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

            // 發送第一次請求
            async function startStream() {
                // 清空 Log
                document.getElementById('customLog').innerHTML = '';
                document.getElementById('updatesLog').innerHTML = '';
                document.getElementById('messagesLog').innerHTML = '';
                
                // 產生全新的 thread_id
                currentThreadId = crypto.randomUUID();
                const input = document.getElementById('userInput').value;
                
                await processStream({
                    input: input,
                    thread_id: currentThreadId
                });
            }

            // 發送中斷恢復請求 (Resume)
            async function resumeStream() {
                // 隱藏審核控制台，恢復發送區
                document.getElementById('interruptArea').style.display = 'none';
                document.getElementById('actionArea').style.display = 'block';
                
                // 取得工程師修改的參數 (將字串轉回 JSON Array)
                const rawInput = document.getElementById('resumeInput').value;
                let resumeData;
                try {
                    resumeData = JSON.parse(rawInput);
                } catch (e) {
                    alert("參數格式錯誤，請輸入有效的 JSON 陣列，例如: [\\"A\\", \\"B\\"]");
                    document.getElementById('interruptArea').style.display = 'block';
                    document.getElementById('actionArea').style.display = 'none';
                    return;
                }

                // 在 Log 中印出人類介入的提示
                document.getElementById('updatesLog').innerHTML += `<span style="color: #28a745; font-weight: bold;">[前端] 👨‍💻 傳送審核結果：${rawInput}</span><br>`;

                // 帶著原來的 thread_id 發送 resume 指令
                await processStream({
                    resume: resumeData,
                    thread_id: currentThreadId
                });
            }
        </script>
    </body>
    </html>
    """
    return html_content

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)