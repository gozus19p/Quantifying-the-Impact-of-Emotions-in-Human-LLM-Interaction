from fastapi import FastAPI
import uvicorn
from generation.velvet import generate, ChatIteration, ChatResponse


app = FastAPI()


@app.post("/api/v1/chat")
def chat(iteration: ChatIteration) -> ChatResponse:
    answer: str = generate(iteration)
    return ChatResponse(message=answer)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
