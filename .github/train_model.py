import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import os

def train_and_convert():
    # 1. 데이터 수집 (삼성전자를 예시로 하되, 데이터를 충분히 확보)
    symbol = "005930.KS"
    print(f"{symbol} 데이터 다운로드 중...")
    data = yf.download(symbol, start="2022-01-01")
    
    if data.empty:
        print("데이터를 가져오지 못했습니다.")
        return

    # 2. 데이터 전처리
    df = data[['Open', 'Close']].copy()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    X, y = [], []
    window_size = 5 # 5일치 데이터 사용
    
    for i in range(window_size, len(scaled_data)):
        # 5일간의 시가/종가 (총 10개 값)를 1차원 배열로 펼침
        X.append(scaled_data[i-window_size:i].flatten())
        # 오늘의 종가를 정답으로 설정
        y.append(scaled_data[i, 1])
    
    X, y = np.array(X), np.array(y)

    # 3. 신경망 모델 구성 (nnet 스타일의 MLP)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    
    # 4. 모델 학습
    print("모델 학습 시작...")
    model.fit(X, y, epochs=50, batch_size=16, verbose=0)

    # 5. TFLite 변환
    print("TFLite 변환 중...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # 6. 파일 저장
    with open('kospi_model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("kospi_model.tflite 생성 완료!")

if __name__ == "__main__":
    train_and_convert()