import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import yfinance as yf


print(" ,---.   ,--.               ,--.                                    ")
print("'   .-',-'  '-. ,---.  ,---.|  |,-.                                 ")
print("`.  `-.'-.  .-'| .-. || .--'|     /                                 ")
print(".-'    | |  |  ' '-' '\ `--.|  \  \                                 ")
print("`-----'  `--'   `---'  `---'`--'`--'                                ")
print(",------.                  ,--.,--.        ,--.  ,--.                ")
print("|  .--. ',--.--. ,---.  ,-|  |`--' ,---.,-'  '-.`--' ,---. ,--,--,  ")
print("|  '--' ||  .--'| .-. :' .-. |,--.| .--''-.  .-',--.| .-. ||      \ ")
print("|  | --' |  |   \   --.\ `-' ||  |\ `--.  |  |  |  |' '-' '|  ||  | ")
print("`--'     `--'    `----' `---' `--' `---'  `--'  `--' `---' `--''--' ")
print(",--------.             ,--.                                         ")
print("'--.  .--',---.  ,---. |  |                                         ")
print("   |  |  | .-. || .-. ||  |                                         ")
print("   |  |  ' '-' '' '-' '|  |                                         ")
print("   `--'   `---'  `---' `--'                                         ")
print("\n\n")
print("Made by Ja3c Ver 1.0")
print("\n\n")
print("Disclaimer: 해당 모델은 선형 회귀 (Linear Regression) 모델을 사용하여 주식 가격을 예측하는 예시입니다. 간단한 추세를 파악하는것 및 가격예측을 해주는 툴 이기에 재미용으로만 쓰세요 \n 해당툴은 리딩방이 아닙니다.")
print("\n\n")

# 주식 이름 받기
stock_name = input("주식 종목 심볼 (예시: Apple 가격예측을 하고 싶으면 AAPL을 적어주세요: ")
print("\n")
# 주식 데이터 불러오기
stock_symbol = stock_name
# 주식 학습데이터 불러오기
print("학습시킬 데이터를 불러와야 합니다. 학습시킬 데이터 날짜 (시작+끝)들을 입력해주세요")
start_input = input("해당종목의 학습시킬 시작날짜를 적어주세요 (Ex) 입력예시: YYYY-MM-DD ")
end_input = input("해당 종목의 학습시킬 종료날짜를 적어주세요 (Ex) 입력예시: YYYY-MM-DD ")

print("==========================학습데이터 불러오는중=======================")
start_date = start_input
end_date = end_input
data = yf.download(stock_symbol, start=start_date, end=end_date)

# 'Close' 컬럼을 기반으로 데이터 전처리
data['Price'] = data['Close']
data = data[['Price']]

# Feature와 Target 데이터 준비
data['Prediction'] = data['Price'].shift(-1)
X = np.array(data.drop(['Prediction'], 1))
X = X[:-1]
y = np.array(data['Prediction'])
y = y[:-1]

# 데이터 분할 (학습 데이터와 테스트 데이터)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 선형 회귀 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)
print("==================================결과===============================")
# 모델 평가
accuracy = model.score(X_test, y_test)
accuracy = accuracy * 100
print("\n")
print(f'모델 예측 정확도: {accuracy} % ')

# 마지막 날의 가격을 예측
last_price = data.iloc[-1]['Price']
predicted_price = model.predict([[last_price]])
print(f'모델 예측 가격: {predicted_price}')
