import os
import re
import json
import subprocess
from docx import Document


#-----------------------------
# 读取 docx 文件
#-----------------------------
def read_docx(path):
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs])


#-----------------------------
# 读取 doc 文件（通过 antiword）
#-----------------------------
def read_doc(path):
    try:
        text = subprocess.check_output(["antiword", path], stderr=subprocess.STDOUT)
        return text.decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"antiword 读取失败: {path}, 错误: {e}")
        return None


#-----------------------------
# 清洗文本：只从“第一条”开始
#-----------------------------
def trim_before_first_article(text):

    # “第一条、第一章、第一节”等都可能出现
    m = re.search(r"第[一二三四五六七八九十百千0-9]+条", text)
    if m:
        return text[m.start():]

    # 再退一步，从 “第一章” 开始
    m = re.search(r"第[一二三四五六七八九十百千0-9]+章", text)
    if m:
        return text[m.start():]

    return text  # 没找到就不裁剪


#-----------------------------
# 判断是否是章节行（需过滤）
#-----------------------------
def is_chapter_line(line):
    # 匹配：第二章、第一章 总则
    if re.match(r"第[一二三四五六七八九十百千0-9]+章", line.strip()):
        return True
    return False


#-----------------------------
# 主条文提取（增强版）
#-----------------------------
def extract_law_items(text):

    text = trim_before_first_article(text)

    lines = text.split("\n")
    law_dict = {}

    current_title = None
    buffer = []

    article_pattern = re.compile(r"^(第[一二三四五六七八九十百千0-9]+条)")

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # 如果是章节标题 → 跳过
        if is_chapter_line(line):
            continue

        # 判断是否为新的条
        m = article_pattern.match(line)
        if m:
            # 如果已有条目，写入
            if current_title:
                law_dict[current_title] = "\n".join(buffer).strip()

            # 开启新条
            current_title = m.group(1)
            buffer = []

            # 去掉条标题，留下真正内容
            content = line[len(current_title):].strip(" ：:　")
            if content:
                buffer.append(content)

        else:
            # 普通内容行
            if current_title:
                buffer.append(line)

    # 最后一条写入
    if current_title:
        law_dict[current_title] = "\n".join(buffer).strip()

    return law_dict



#-----------------------------
# 批处理主程序
#-----------------------------
def process_documents(data_folder="data", output_folder="json_data"):

    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(data_folder):
        filepath = os.path.join(data_folder, file)

        if not (file.endswith(".doc") or file.endswith(".docx")):
            continue

        prefix = file.split("_")[0]

        # 读取文本
        if file.endswith(".docx"):
            text = read_docx(filepath)
        else:
            text = read_doc(filepath)
            if not text:
                print(f"跳过无法解析的 doc 文件: {file}")
                continue

        # 解析条文
        law_items = extract_law_items(text)
        if not law_items:
            print(f"跳过无法律条款的文件: {file}")
            continue

        # 输出 JSON
        output_path = os.path.join(output_folder, f"{prefix}.json")

        json_data = [{f"{prefix} {k}": v for k, v in law_items.items()}]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)

        print(f"✔ 已转换：{file} → {output_path}")


if __name__ == "__main__":
    process_documents()
