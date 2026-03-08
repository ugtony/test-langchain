async function chatWithFactoryAI() {
    try {
        const response = await fetch("http://localhost:8000/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            // 這裡帶入測試用的參數
            body: JSON.stringify({ input: "請協助分析最新蝕刻製程的 Recipe 參數" })
        });

        if (!response.body) throw new Error("ReadableStream not yet supported in this browser.");

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            // 解碼收到的二進位資料
            buffer += decoder.decode(value, { stream: true });
            
            // SSE 的資料區塊是用兩個換行符號分隔的
            const chunks = buffer.split("\n\n");
            
            // 留下最後一個不完整的區塊在 buffer 中，等待下一次讀取
            buffer = chunks.pop() || "";

            for (const chunk of chunks) {
                // 過濾掉空白行，並移除 "data: " 前綴
                if (chunk.trim() !== "" && chunk.startsWith("data: ")) {
                    const jsonString = chunk.substring(6); // 移除 "data: "
                    try {
                        const parsedData = JSON.parse(jsonString);
                        
                        // 根據 Server 端設計的 mode 進行畫面更新
                        if (parsedData.mode === "custom") {
                            console.log(`[進度更新] 來自子圖 ${parsedData.namespace}:`, parsedData.chunk.data.msg);
                        } else if (parsedData.mode === "updates") {
                            console.log(`[狀態更新]`, parsedData.chunk);
                        } else if (parsedData.mode === "error") {
                            console.error(`[系統錯誤]`, parsedData.msg);
                        }
                    } catch (e) {
                        console.error("JSON 解析失敗，原始字串為:", jsonString);
                    }
                }
            }
        }
    } catch (error) {
        console.error("請求發送失敗:", error);
    }
}

// 執行測試
chatWithFactoryAI();