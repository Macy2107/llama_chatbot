"""Entry point: right-click > Run 'run' in PyCharm, then open http://127.0.0.1:8000"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
