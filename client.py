import asyncio
import json
import httpx
import uuid

async def stream_request(url: str, payload: dict):
    """將串流請求封裝成可重複使用的函數"""
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", url, json=payload, timeout=None) as response:
            async for line in response.aiter_lines():
                if not line.startswith("data: "): continue
                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                mode = data.get("mode")
                chunk = data.get("chunk")
                ns_str = "/".join(data.get("namespace", [])) or "root"

                if mode == "error":
                    print(f"❌ [系統錯誤] {data.get('msg')}")
                elif mode == "custom":
                    print(f"⏳ [{ns_str} - 進度] {chunk.get('msg')}")
                elif mode == "updates":
                    # 當圖被 interrupt 時，updates 模式會吐出特殊的 __interrupt__ 標記
                    if "__interrupt__" in chunk:
                        print(f"✋ [{ns_str} - 觸發中斷] 系統已凍結，等待人類指令...")
                    else:
                        print(f"🔄 [{ns_str} - 狀態更新] {json.dumps(chunk, ensure_ascii=False)}")
                elif mode == "messages":
                    print(f"💬 [{ns_str} - 對話] {chunk.get('type')}: {chunk.get('content')}")

async def run_client():
    url = "http://localhost:8000/chat"
    
    # 🌟 關鍵：必須記住這個 thread_id，才能接續中斷的對話
    session_thread_id = str(uuid.uuid4())

    print("=========================================")
    print("🚀 第一階段：啟動分析 (預期會在審核節點停下)")
    print("=========================================")
    payload_1 = {
        "input": "請協助分析最新蝕刻製程的 Recipe 參數",
        "thread_id": session_thread_id
    }
    await stream_request(url, payload_1)

    print("\n=========================================")
    print("👨‍💻 人類工程師介入中...")
    print("（假設工程師將 Gas Flow 剔除，並加入了 Temperature）")
    await asyncio.sleep(2) # 模擬思考時間
    print("=========================================\n")

    print("=========================================")
    print("🚀 第二階段：傳送修改後的參數，喚醒圖形繼續執行")
    print("=========================================")
    # 🌟 關鍵：帶入 resume 欄位，並且 thread_id 要跟剛剛一樣
    payload_2 = {
        "resume": ["Temperature", "RF Power", "Pressure"], 
        "thread_id": session_thread_id
    }
    await stream_request(url, payload_2)
    
    print("\n✅ 整個流程順利走完！")

if __name__ == "__main__":
    asyncio.run(run_client())