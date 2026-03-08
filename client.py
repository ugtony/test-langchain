import asyncio
import json
import httpx

async def run_client():
    url = "http://localhost:8000/chat"
    payload = {
        "input": "請協助分析最新蝕刻製程的 Recipe 參數"
    }

    print(f"🚀 發送請求至 {url}...\n")
    print("-" * 50)

    # 建立非同步 HTTP 客戶端
    async with httpx.AsyncClient() as client:
        # 使用 POST 發送請求，並開啟 stream 模式接收 SSE
        # timeout=None 確保長時間運算的圖不會中途斷線
        async with client.stream("POST", url, json=payload, timeout=None) as response:
            
            # 逐行讀取伺服器回傳的串流資料
            async for line in response.aiter_lines():
                # 過濾掉空白行，只處理 SSE 標準的 "data: " 開頭
                if not line.startswith("data: "):
                    continue

                # 移除 "data: " 前綴，提取純 JSON 字串
                json_str = line[6:]
                
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    print(f"⚠️ 解析失敗的字串: {json_str}")
                    continue

                mode = data.get("mode")
                chunk = data.get("chunk")
                
                # 將 namespace 陣列轉為可讀的字串，例如 "clarify/research"
                namespace = data.get("namespace", [])
                ns_str = "/".join(namespace) if namespace else "root"

                # 根據不同模式，印出不同格式的 Log
                if mode == "error":
                    print(f"❌ [系統錯誤] {data.get('msg')}")
                elif mode == "custom":
                    print(f"⏳ [{ns_str} - 進度] {chunk.get('msg')}")
                elif mode == "updates":
                    # 將 dict 轉回字串印出，方便閱讀
                    chunk_str = json.dumps(chunk, ensure_ascii=False)
                    print(f"🔄 [{ns_str} - 狀態更新] {chunk_str}")
                elif mode == "messages":
                    print(f"💬 [{ns_str} - 對話] {chunk.get('type')}: {chunk.get('content')}")

    print("-" * 50)
    print("✅ 串流接收完畢！")

if __name__ == "__main__":
    # 執行非同步主程式
    asyncio.run(run_client())