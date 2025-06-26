# 本代码用于二次整理爬取的内容。

with open('all.txt', 'r') as file:
    s = file.read()
lst ='#'
s += '#'
S=[]
T=[]
ss=[]
for i in range(len(s)-1) :
    if s[i] =='#' and lst !='#'and s[i+1] =='#':
        S.append("".join(T))
        T=[]
    T.append(s[i])
    lst = s[i]
for t in S:
    if (len(t)>=7 and t[4]== ' ' and t[5] =='代' and t[6]=='码'):
        continue
    elif (len(t)>=6 and t[3]== ' ' and t[4] =='代' and t[5]=='码'):
        continue 
    elif (len(t)>=9 and t[5]== '输' and t[7] =='格' and t[8]=='式'):
        continue 
    elif (len(t)>=8 and t[4]== '输' and t[6] =='格' and t[7]=='式'):
        continue 
    else:
        ss.append(t)

S="".join(ss)
final={}
for c in S:
    try:
        c.encode('utf-8')
        final.append(c)
    except UnicodeEncodeError:
        # 捕获编码错误，跳过非法字符
        pass


with open('./all_solution.txt', 'w') as file:
    file.write("".join(ss))