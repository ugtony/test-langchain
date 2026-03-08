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
# 1. 狀態定義 (子圖也必須繼承 messages)
# ==========================================
class SubState(TypedDict):
    terms: List[str]
    # 🌟 讓子圖也能讀寫對話紀錄，這樣才能記錄工程師的提問與 AI 的回答
    messages: Annotated[list[BaseMessage], add_messages]

class MainState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    terms: List[str]

# ==========================================
# 2. 建立子圖 (回到起點的迴圈設計)
# ==========================================
async def research_node(state: SubState, writer: StreamWriter):
    # 🌟 動態抓取對話紀錄的最後一句話，確認目前要分析的目標
    messages = state.get("messages", [])
    current_query = messages[-1].content if messages else "未知指令"
    
    # 根據最新的提問重新啟動掃描
    writer({"node": "research_node", "msg": f"收到指示「{current_query}」，正在重新掃描機台感測器與歷史紀錄..."})
    await asyncio.sleep(1.0) 
    
    # 實務上這裡會呼叫 RAG 或 LLM 根據 current_query 產出不同的 terms
    return {"terms": ["Gas Flow", "RF Power", "Pressure"]}

async def human_review_node(state: SubState, writer: StreamWriter):
    writer({"node": "human_review_node", "msg": f"抓取到預設參數：{state.get('terms')}，準備暫停等待確認..."})
    
    human_feedback = interrupt("請確認參數 (輸入 JSON)，或直接輸入新指示讓系統重新分析：")
    
    # 1. 嘗試解析是否為合格的參數 (JSON Array)
    try:
        parsed = json.loads(human_feedback) if isinstance(human_feedback, str) else human_feedback
        if isinstance(parsed, list):
            writer({"node": "human_review_node", "msg": f"✅ 收到有效參數：{parsed}，審核通過。"})
            return {"terms": parsed}
    except Exception:
        pass # 解析失敗，不報錯，繼續往下走

    # 🌟 2. 解析失敗 -> 當作全新問題，跳回起點！
    writer({"node": "human_review_node", "msg": f"偵測到新問題，系統將回到起點重新啟動分析流程..."})
    
    # 使用 Command 把新問題加入對話，並強制跳回 "research" 節點
    return Command(
        goto="research",
        update={"messages": [HumanMessage(content=str(human_feedback))]}
    )

sub_builder = StateGraph(SubState)
sub_builder.add_node("research", research_node)
sub_builder.add_node("review", human_review_node)

sub_builder.add_edge(START, "research")
sub_builder.add_edge("research", "review")
# 如果 review 成功解析參數，走預設路徑結束子圖，回到主圖的 llm 節點
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
            <p style="color: #666; font-size: 14px;">請修改下方陣列 (JSON 格式)，<b>或直接輸入新指令讓系統重新分析 (例如: 改幫我查前天機台當機時的歷史紀錄)</b>：</p>
            <input type="text" id="resumeInput" value='["Temperature", "RF Power", "Pressure"]' />
            <button class="btn-success" onclick="resumeStream()">送出</button>
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
                document.getElementById('interruptArea').style.display = 'none';
                document.getElementById('actionArea').style.display = 'block';
                
                // 🌟 直接取得原始字串，不再用 JSON.parse() 強制檢查！
                const rawInput = document.getElementById('resumeInput').value;

                document.getElementById('updatesLog').innerHTML += `<span style="color: #28a745; font-weight: bold;">[前端] 👨‍💻 傳送審核結果或提問：${rawInput}</span><br>`;

                // 將原汁原味的字串傳給後端，讓後端的 try-except 去判斷
                await processStream({
                    resume: rawInput,
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