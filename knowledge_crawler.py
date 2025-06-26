# 本代码基于 AI 生成的代码整合，可以输入一些题目信息，爬取对应的内容。
# 其中爬取某一特定网页内容的代码来源于 AI。

import requests
import os
from bs4 import BeautifulSoup

import html2text


def convert_html_to_markdown(html_content):
    """将 HTML 内容转换为 Markdown 并保留数学公式"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 处理行内数学公式（假设使用 <span class="math inline">）
    for element in soup.find_all('span', class_='math inline'):
        latex = element.get_text(strip=True)
        element.replace_with(f'${latex}$')
    
    # 处理块级数学公式（假设使用 <div class="math display">）
    for element in soup.find_all('div', class_='math display'):
        latex = element.get_text(strip=True)
        element.replace_with(f'$$\n{latex}\n$$')
    
    # 配置 HTML 转 Markdown 转换器
    converter = html2text.HTML2Text()
    converter.body_width = 0          # 禁用自动换行
    converter.ignore_links = False    # 保留链接
    converter.ignore_images = False   # 保留图片
    
    return converter.handle(str(soup))


def rebuild(s):
    t = ""
    for c in s :
        if(c == ';'):
            t += '%3B'
        else:
            t += c
    return t
markdown = ""
while 1:
    s=input()
    if (s == "end"):
        break
    v=[""]
    n=0
    for c in s:
        if(c=='\t'):
            v.append("")
            n = n + 1
        else:
            v[n] += c
    n = n + 1
    if(n!=7 or v[6] == "(Encrypted)"):
        continue
    url = 'https://yhx-12243.github.io/OI-transit/records/'+ rebuild(v[1])+'.html'
    response = requests.get(url)
    if response.status_code == 200:
        # 使用BeautifulSoup解析HTML内容
        soup = BeautifulSoup(response.text, 'html.parser')
    
        # 获取整个页面的HTML源代码
        markdown += convert_html_to_markdown(soup.prettify())
        markdown += "\n###思维难度 : " + v[4]
        markdown += "\n###代码难度 : " + v[5]
        markdown += "\n###所用知识点 : " + v[6]
        markdown += '\n\n---\n\n'
    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
final = []

for c in markdown:
    try:
        c.encode('utf-8')
        final.append(c)
    except UnicodeEncodeError:
        # 捕获编码错误，跳过非法字符
        pass
with open('./all.txt', 'w') as file:
    file.write("".join(final))
