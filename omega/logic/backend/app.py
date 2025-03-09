from fastapi import FastAPI
import uvicorn
from generation.velvet import generate_streaming, ChatIteration
from fastapi.responses import StreamingResponse


app = FastAPI()


@app.post("/api/v1/chat")
def streaming_chat(iteration: ChatIteration) -> str:
    return StreamingResponse(generate_streaming(iteration), media_type="text/plain")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
