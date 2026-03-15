import io
import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

app = FastAPI()

#Загружаем модель один раз при старте приложения для экономии ресурсов
model = joblib.load('model.joblib')

@app.post('/predict')
async def predict_from_csv(file: UploadFile = File(...)):
    #Проверка формата файла
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail='Файл должен быть в формате CSV')

    #Чтение содержимого файла в память
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    #Выполнение предсказания
    try:
        predictions = model.predict_proba(df)[:, 1]
        predictions = (predictions >= 0.2).astype(int)
        df['prediction'] = predictions
        result = df[['id', 'prediction']]
        #Сохраняем результат в виртуальный файл
        stream = io.StringIO()
        result.to_csv(stream, index=False)
        
        #Перематываем "виртуальный файл" в начало
        response = io.BytesIO(stream.getvalue().encode())
        
        #Возвращаем файл пользователю
        return StreamingResponse(
            response,
            media_type="text/csv",
            headers={'Content-Disposition': f'attachment; filename=result_{file.filename}'}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Ошибка при предсказании: {str(e)}')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)