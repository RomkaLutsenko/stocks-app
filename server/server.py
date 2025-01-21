from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess

# Создаем экземпляр приложения
app = FastAPI()

# Определяем модель для передачи данных
class ScriptInput(BaseModel):
    script_name: str
    args: list[str]

@app.post("/execute")
async def execute_script(data: ScriptInput):
    """
    Выполняет указанный Python-скрипт с переданными аргументами.
    """
    try:
        # Выполнение скрипта
        result = subprocess.run(
            ["python3", data.script_name] + data.args,
            capture_output=True,
            text=True,
        )

        # Проверяем на ошибки
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=result.stderr)

        return {"output": result.stdout}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
