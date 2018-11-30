'''
엑셀 파일을 사용하기 위해서 openpyxl 패키지를 먼저 설치해야한다.
$ pip install openpyxl
'''

import openpyxl
 
# 엑셀 파일 열기
wb = openpyxl.load_workbook('test.xlsx')
 
# Sheet 선택
ws = wb["Sheet1"]
 
# 모든 데이터를 입력받아서 row 단위로 화면에 출력하기
for row in ws.rows: # 모든 row에 대하여
    for cell in row:
        print(cell.value, end=' ')
    print()

# 모든 데이터를 입력받아서 row 단위로 list로 관리하기
for row in ws.rows: # 모든 row에 대하여
    l = []
    for cell in row:
        l.append(cell.value)
    print(l)
    
# 텐서플로우의 입력값으로 사용하기 위해서 list of list로 만들기
# 
train_data = []
train_label = []
for row in ws.rows: # 모든 row에 대하여
    one_row = []
    for cell in row:
        if cell.column == 'A':
            train_label.append([cell.value])
        else:
            one_row.append(cell.value)
    train_data.append(one_row)

print(train_data)
print(train_label)
    
# 엑셀 파일 닫기 
wb.close()