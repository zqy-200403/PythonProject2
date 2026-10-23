# main.py —— AI一对一家教：选老师 -> 聊天 -> 代码验证答案锚点 -> 自动判卷 -> 质疑持锚 -> 错题本 -> 时间感知 -> 保存
# 核心原则：标准答案一律由本地 sympy 代码算出，LLM 只负责讲解，不负责算数
# 命名规范：代码标识符统一英文 snake_case；注释、提示词保持中文

import os
import re
import json
import time
import math
import random
import difflib
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

from teachers import TEACHERS

try:
    import sympy
    SYMPY_OK = True
except ImportError:
    SYMPY_OK = False

load_dotenv()
client = OpenAI(
    api_key=os.getenv("GLM_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
)

# ================= 配置 =================
TEACHER_MODEL = "glm-4-flash"    # 老师模型：只负责聊天讲解
SOLVER_MODEL = "glm-4.5-flash"   # 解题/判卷模型：更强，失败时自动回退
TEACHER_TEMPERATURE = 0.3        # 教学场景要准确，调低发散度
TYPING_SPEED = 0.01              # 流式打印的微停顿；设为 0 = 跟随网络节奏

WRONG_BOOK_FILE = "错题本.json"
LAST_TIME_FILE = "上次学习时间.txt"

# ================= 业务数据 =================
KNOWLEDGE = {
    "射影定理": (
        "直角三角形中，斜边上的高是两条直角边在斜边上射影的比例中项。"
        "特别注意：p、q 是【斜边被高分成的两段】（射影长），不是两条直角边！"
        "公式：h² = p·q；a² = p·c，b² = q·c（a、b为直角边，c为斜边）。"
        "已知两直角边 a、b 和斜边 c 时，斜边上的高直接用面积法求：h = a×b÷c。"
    ),
    "勾股定理": "直角三角形两直角边 a、b 与斜边 c 满足：a² + b² = c²。",
    "勾股定理逆定理": "若三角形三边满足 a² + b² = c²，则该三角形是直角三角形，c 所对的角是直角。",
    "圆柱体积": "圆柱体积 = 底面积 × 高，即 V = πr²h（r为底面半径，h为高）。",
    "圆的面积": "圆的面积 S = πr²，周长 C = 2πr。",
    "三角形全等": (
        "判定：SSS（三边相等）、SAS（两边及其夹角）、ASA（两角及其夹边）、"
        "AAS（两角及其中一角的对边）、HL（直角三角形的斜边和一条直角边）。"
    ),
    "相似三角形": "对应角相等、对应边成比例。判定：两角对应相等（AA）；两边对应成比例且夹角相等；三边对应成比例。",
    "牛顿第二定律": "F = ma：物体加速度与所受合外力成正比，与质量成反比。",
    "密度公式": "密度 = 质量 ÷ 体积，即 ρ = m/V。",
    "压强公式": "压强 p = F/S；液体内部压强 p = ρgh。",
    "欧姆定律": "I = U/R：导体中的电流与电压成正比、与电阻成反比。",
    "速度公式": "v = s/t：速度 = 路程 ÷ 时间。",
}

CHALLENGE_WORDS = ("不对", "错了", "有错", "是不是错", "有问题", "质疑",
                   "确定吗", "真的吗", "应该不是", "搞错", "不太对", "好像不对")
ASSERT_WORDS = ("正确答案", "答案应该", "应该是", "应为")
HELP_WORDS = ("不知道", "不会", "不清楚", "忘了", "没学过", "教我",
              "不明白", "不懂", "听不懂", "再讲", "讲解", "检查", "继续")
NEGATION_WORDS = ("不是", "不等于", "并非", "≠")

PYTHAGOREAN_TRIPLES = [(3, 4, 5), (6, 8, 10), (5, 12, 13), (9, 12, 15),
                       (8, 15, 17), (7, 24, 25), (20, 21, 29)]

TEACHING_RULES = """

【教学立场规则（最高优先级，必须遵守）】
1. 一切答案对错以系统提供的【标准答案】为唯一依据——它由程序计算验证，绝对权威。你只负责讲解，禁止自己另算答案。
2. 学生质疑或主张不同答案时：不得改口。先确认学生说的是哪道题，然后一步步展示推导，引导学生自己发现错在哪。
3. 严禁编造不存在的定理、公式或概念；讲解时严格依据【权威知识点】（若系统提供）。
4. 不要问"你对这个答案满意了吗"这类讨好式问句；确认学生理解后直接推进。
5. 教学权威来自正确，不来自顺从；语气始终保持温和耐心。
"""


# ================= LaTeX 清洗与流式输出 =================
def clean_formula(text):
    """把常见 LaTeX 符号和 Markdown 标记转成终端友好的纯文本"""
    text = text.replace("\\[", "").replace("\\]", "")
    text = text.replace("\\(", "").replace("\\)", "").replace("$", "")
    text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\\sqrt\{([^{}]+)\}", r"√(\1)", text)
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"_\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"_([a-zA-Z0-9])", r"\1", text)
    symbol_map = {
        "\\times": "×", "\\cdot": "·", "\\pi": "π", "\\approx": "≈",
        "\\pm": "±", "\\leq": "≤", "\\geq": "≥", "\\neq": "≠",
        "\\Rightarrow": "→", "\\rightarrow": "→",
        "\\left": "", "\\right": "",
        "\\alpha": "α", "\\beta": "β", "\\theta": "θ", "\\Delta": "Δ",
    }
    for old, new in symbol_map.items():
        text = text.replace(old, new)
    text = re.sub(r"\^\{?2\}?", "²", text)
    text = re.sub(r"\^\{?3\}?", "³", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"(?m)^#{1,6}\s+", "", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    return text


def _braces_closed(text, count):
    """检查形如 {...}{...} 的文本是否已凑齐指定数量的闭合组"""
    depth = 0
    closed = 0
    for char in text:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                closed += 1
                if closed == count:
                    return True
    return False


def is_mature(chunk):
    """判断扣留的片段是否已完整、可以清洗输出了"""
    if len(chunk) > 80:
        return True
    if chunk.startswith("\\"):
        match = re.match(r"\\([a-zA-Z]+)", chunk)
        if not match:
            return len(chunk) >= 2
        command = match.group(1)
        rest = chunk[1 + len(command):]
        if command in ("frac", "sqrt", "text", "mathrm"):
            need = 2 if command == "frac" else 1
            return bool(rest) and _braces_closed(rest, need)
        return len(rest) >= 1 and not rest[0].isalpha()
    if chunk.startswith("**"):
        return chunk.count("**") >= 2
    if chunk.startswith("*"):
        return len(chunk) >= 2
    if chunk.startswith("^") or chunk.startswith("_"):
        if len(chunk) < 2:
            return False
        if chunk[1] == "{":
            return _braces_closed(chunk[1:], 1)
        return True
    if chunk.startswith("$"):
        return len(chunk) >= 2
    return True


def extract_printable(pending):
    """拆出确定安全的前缀立即打印，疑似符号残片继续扣留。返回 (可打印, 扣留)"""
    positions = [pending.find(c) for c in ("\\", "*", "^", "$", "_") if c in pending]
    if not positions:
        return pending, ""
    start = min(positions)
    safe, chunk = pending[:start], pending[start:]
    if is_mature(chunk):
        return safe + clean_formula(chunk), ""
    return safe, chunk


def stream_reply(teacher_name, messages):
    """边收边打印；疑似 LaTeX 的片段扣住凑完整再清洗输出。返回清洗后的全文"""
    stream = client.chat.completions.create(
        model=TEACHER_MODEL,
        messages=messages,
        temperature=TEACHER_TEMPERATURE,
        stream=True,
    )
    print(f"\n{teacher_name}：", end="", flush=True)
    reply = ""
    chunk = ""
    for event in stream:
        if not event.choices:
            continue
        piece = event.choices[0].delta.content or ""
        if not piece:
            continue
        reply += piece
        chunk += piece
        printable, chunk = extract_printable(chunk)
        if printable:
            print(printable, end="", flush=True)
            if TYPING_SPEED:
                time.sleep(TYPING_SPEED)
    if chunk:
        print(clean_formula(chunk), end="", flush=True)
    print()
    return clean_formula(reply)


# ================= 模型调用底层 =================
def call_model(prompt, models=(SOLVER_MODEL, TEACHER_MODEL)):
    """非流式调用：依次尝试模型列表，全部失败则抛出最后一个异常"""
    last_error = None
    for model_name in models:
        try:
            resp = client.chat.completions.create(
                model=model_name,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content
        except Exception as error:
            last_error = error
    raise last_error


def parse_json(raw):
    """从模型输出中提取最外层 JSON 对象，失败返回 None"""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return json.loads(match.group())
    return None


def format_number(value):
    """浮点数转干净字符串：整数去掉小数点，其余保留 6 位有效数字"""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6g}"


# ================= 答案锚点：LLM 写代码，本地 sympy 计算 =================
def solve_and_verify(question):
    """让解题模型写 sympy 代码，本地执行得到答案；LLM 不直接给出数值"""
    if not SYMPY_OK:
        return None
    prompt = (
        "解下面这道初中数学题：先想清楚求什么，再写Python代码计算。\n"
        f"【题目】{question}\n"
        "只输出一个Python代码块（python 开头），要求：\n"
        "1. 第一行 from sympy import *\n"
        "2. 最终答案存入变量 answer（能精确就精确，如 Rational(12,5)、sqrt(12)）\n"
        "3. 不要 input()，不要 print，不要输出解释文字"
    )
    try:
        raw = call_model(prompt)
    except Exception as error:
        print(f"\n[解题服务暂时不可用：{type(error).__name__}]")
        return None
    match = re.search(r"(?:python)?\s*(.*?)", raw, re.DOTALL)
    if match:
        code = match.group(1)
    else:
        code = raw
    if ("answer" not in code) or ("input(" in code):
        return None
    try:
        namespace = {}
        exec(code, namespace)  # 本地自用工具：代码由解题模型生成，已过滤 input
        answer_obj = namespace.get("answer")
        if answer_obj is None:
            return None
        try:
            value = float(answer_obj)
        except Exception:
            value = None
        if value is not None:
            answer_text = format_number(value)
        else:
            answer_text = str(answer_obj)
        return {"answer_text": answer_text, "value": value}
    except Exception:
        return None


def lock_question(teacher_reply):
    """两步锁定标准答案：
    ① 抽题并重写成独立题面（严禁携带老师此前的推导结论，防污染）
    ② sympy 代码求解验证；验证失败就不设锚、不判卷——宁可缺席，不再出错"""
    try:
        raw = call_model(
            "下面是一位老师发给学生的最后一条消息。判断结尾是否向学生提出了"
            "需要给出具体知识性答案的问题。\n"
            f"【老师消息】{teacher_reply}\n"
            '只输出JSON：{"is_question": true或false, "type": "knowledge"或"process", '
            '"question": "题面原文"}\n'
            "type判定：若问的是\"下一步做什么/明白了吗/对不对/满意吗\"等过程性或确认性问题，"
            "type为process；只有需要学生给出知识性答案的才是knowledge。"
            "若type为knowledge，请把题面重写为【独立可解】的完整题目，"
            "只保留题目本身的条件，严禁引用老师消息中任何中间计算结果或结论。"
            "若is_question为false，question留空。JSON外不要有任何文字。"
        )
        data = parse_json(raw)
        if not data:
            return None
        if not (data.get("is_question") and data.get("type") == "knowledge"):
            return None
        question = (data.get("question") or "").strip()
        if not question:
            return None
        result = solve_and_verify(question)
        if result:
            return {
                "question": question,
                "answer": result["answer_text"],
                "value": result["value"],
                "verified": True,
            }
        print("[系统]（本题无法自动验证答案，本轮不判卷）")
    except Exception as error:
        print(f"\n[出题锁定暂时不可用：{type(error).__name__}]")
    return None


# ================= 数值提取与判卷 =================
def extract_numbers(text):
    """从学生回答中提取数值：算式、根式、纯数字"""
    normalized = (
        text.replace("，", " ")
            .replace("。", " ")
            .replace("×", "*")
            .replace("÷", "/")
            .replace("−", "-")
    )
    values = []
    # 1) 纯算式（只含数字与加减乘除，安全求值）：如 3*4/5、12/5
    for expr in re.findall(r"-?\d+(?:\.\d+)?(?:\s*[+\-*/]\s*-?\d+(?:\.\d+)?)+", normalized):
        if re.fullmatch(r"[-\d.\s*/+]+", expr):
            try:
                values.append(float(eval(expr, {"__builtins__": {}}, {})))
            except Exception:
                pass
    # 2) 根式：如 2√3、2倍根号3、√12
    for match in re.findall(
        r"(-?\d+(?:\.\d+)?)?\s*(?:\*|×)?\s*倍?\s*(?:√|根号)\s*\(?\s*(\d+(?:\.\d+)?)",
        normalized,
    ):
        try:
            coeff = float(match[0]) if match[0] else 1.0
            values.append(coeff * math.sqrt(float(match[1])))
        except Exception:
            pass
    # 3) 纯数字兜底
    for num in re.findall(r"-?\d+(?:\.\d+)?", normalized):
        values.append(float(num))
    return values


def numbers_close(a, b):
    """判断两个数值是否等价"""
    return abs(a - b) < 1e-9 or round(a, 2) == round(b, 2)


def grade_answer(anchor, student_answer):
    """代码比对优先；无法数值化时才用 LLM（且只对照锚点）。
    返回 True/False、'skip'（没在答题）或 None（判卷失败）"""
    if anchor.get("value") is not None and not any(w in student_answer for w in NEGATION_WORDS):
        values = extract_numbers(student_answer)
        if values:
            if any(numbers_close(v, anchor["value"]) for v in values):
                return True, f"回答正确（标准答案 {anchor['answer']}）"
            return False, f"与标准答案 {anchor['answer']} 不符"
    try:
        raw = call_model(
            "你是严格的阅卷老师。\n"
            f"【题目】{anchor['question']}\n"
            f"【标准答案（程序验证，绝对权威）】{anchor['answer']}\n"
            f"【学生回答】{student_answer}\n"
            '只输出JSON：{"correct": true或false或"skip", "note": "20字内点评"}。'
            "与标准答案一致→true，不一致→false，"
            "学生没在答题（反问、质疑、闲聊、问新问题）→skip。"
            "数值等价即一致（如12/5与2.4、2√3与3.46）。JSON外不要有任何文字。"
        )
        data = parse_json(raw)
        if not data:
            return None, ""
        if data.get("correct") == "skip":
            return "skip", data.get("note", "学生切换了话题")
        return bool(data.get("correct")), data.get("note", "")
    except Exception as error:
        print(f"\n[判卷服务暂时不可用：{type(error).__name__}]")
        return None, ""


# ================= 题库：代码出题，答案天生正确 =================
def generate_variation():
    """从题库生成练习题，答案由代码算出。返回 (题面, 数值答案)"""
    a, b, c = random.choice(PYTHAGOREAN_TRIPLES)
    if random.random() < 0.5:
        text = f"直角三角形的两条直角边分别为 {a} 和 {b}，斜边为 {c}。求斜边上的高 h。"
        return text, a * b / c
    text = f"直角三角形的两条直角边分别为 {a} 和 {b}，斜边为 {c}。求直角边 {a} 在斜边上的射影长 p。"
    return text, a * a / c


def make_anchor(text, value):
    """用题面和已算好的数值构造锚点"""
    return {
        "question": text,
        "answer": format_number(value),
        "value": value,
        "verified": True,
    }


# ================= 其他工具 =================
def match_knowledge(text):
    """模糊匹配学生输入里的知识点，容忍错别字（'摄影定理'→'射影定理'）"""
    if not text:
        return None
    best, best_ratio = None, 0.0
    for name in KNOWLEDGE:
        if name in text:
            return name
        size = len(name)
        for i in range(max(0, len(text) - size + 1)):
            ratio = difflib.SequenceMatcher(None, text[i:i + size], name).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best = name
    if best_ratio >= 0.7:
        return best
    return None


def save_wrong_question(anchor, student_answer, note):
    """答错的题写入错题本"""
    records = []
    if os.path.exists(WRONG_BOOK_FILE):
        with open(WRONG_BOOK_FILE, "r", encoding="utf-8") as f:
            records = json.load(f)
    records.append({
        "时间": time.strftime("%Y-%m-%d %H:%M"),
        "题目": anchor["question"][:300],
        "标准答案": anchor.get("answer", ""),
        "学生回答": student_answer,
        "点评": note,
    })
    with open(WRONG_BOOK_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def is_challenge(text):
    """判断学生输入是否属于质疑"""
    return any(word in text for word in CHALLENGE_WORDS)


def load_json_list(path):
    """读取 JSON 列表文件，不存在或为空返回空列表"""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ================= 0. 启动自检：粘贴丢段当场暴露 =================
def startup_check():
    sample = clean_formula("C=2\\pi \\times 3^2，A_{侧}")
    for frag in ("C=2π × 3²", "A侧"):
        if frag not in sample:
            print(f"[代码不完整] 清洗函数自检失败，实际得到：{sample}")
            print("说明 main.py 粘贴时丢了内容，请重新完整复制代码后整体替换。")
            raise SystemExit
    if not SYMPY_OK:
        print("[提示] 未安装 sympy，答案自动验证不可用！请先运行：pip install sympy")


startup_check()

# ================= 1. 选老师 =================
print("=== AI 一对一家教 ===")
for key, teacher_info in TEACHERS.items():
    print(f"{key}. {teacher_info['subject']}——{teacher_info['name']}")

while True:
    choice = input("请选择老师（输入数字）：").strip()
    if choice in TEACHERS:
        teacher = TEACHERS[choice]
        break
    print("没有这个选项，请重新输入。")

history = [{"role": "system", "content": teacher["prompt"] + TEACHING_RULES}]
scores = []
anchor = None  # 当前待答题（含程序验证的标准答案），判卷与回应的唯一依据

# ================= 2. 时间感知状态 =================
last_record = None
if os.path.exists(LAST_TIME_FILE):
    with open(LAST_TIME_FILE, "r", encoding="utf-8") as f:
        last_record = f.read()

gap_hint = ""
if last_record:
    gap = datetime.now() - datetime.strptime(last_record, "%Y-%m-%d %H:%M")
    if gap.days >= 1:
        gap_hint = f"学生已经{gap.days}天没来学习了，开场表达对他的挂念，"
    elif gap.seconds // 3600 >= 2:
        gap_hint = "学生几小时前刚来过，可能是遇到难题又回来了，开场问一句是不是卡住了，"

with open(LAST_TIME_FILE, "w", encoding="utf-8") as f:
    f.write(datetime.now().strftime("%Y-%m-%d %H:%M"))

if gap_hint:
    history.append({
        "role": "system",
        "content": f"【当前状态】{gap_hint}请在第一句话中自然体现，之后恢复正常。",
    })

# ================= 3. 复习错题 =================
wrong_book = load_json_list(WRONG_BOOK_FILE)
if wrong_book:
    print(f"\n📖 发现错题本，共 {len(wrong_book)} 道错题：")
    for i, record in enumerate(wrong_book, 1):
        print(f"{i}. [{record.get('时间', '')}] {record.get('点评', '')}")
    if input("输入题号让老师重新讲解（其他键跳过）：").strip().isdigit():
        n = int(input("题号：")) - 1
        if 0 <= n < len(wrong_book):
            record = wrong_book[n]
            old_question = record.get("题目") or record.get("老师问了什么", "")
            review_request = (
                f"老师，我之前这道题做错了。题目是：{old_question}，"
                f"我的回答是：{record.get('学生回答', '')}，错误点是：{record.get('点评', '')}。"
                + (f"正确答案是：{record.get('标准答案', '')}。" if record.get("标准答案") else "")
                + "请针对我的错误重新讲解，再用面积法验证一遍答案。"
            )
            history.append({"role": "user", "content": review_request})

print(f"\n=== {teacher['name']}已上线，输入 q 退出，输入「出题」做练习 ===")

# ================= 4. 聊天循环 =================
while True:
    user_input = input("\n你：").strip()
    if user_input == "q":
        break
    if not user_input:
        continue

    # 「出题」命令：题库出题，答案已由代码预先算好
    if user_input == "出题":
        question_text, answer_value = generate_variation()
        anchor = make_anchor(question_text, answer_value)
        print("[系统]（练习题已生成，标准答案已由程序锁定）")
        history.append({
            "role": "system",
            "content": (
                f"【变式练习】请把下面这道题出给学生（可轻微改述，切勿透露答案）：\n{question_text}\n"
                f"（系统已验证标准答案：{anchor['answer']}，仅你可见）"
            ),
        })
        history.append({"role": "user", "content": "老师，请给我出一道练习题。"})
        try:
            reply = stream_reply(teacher["name"], history)
            history.append({"role": "assistant", "content": reply})
        except Exception as error:
            print("\n[网络错误]", type(error).__name__)
        continue

    injections = []
    preset_anchor = None

    # ① 质疑/断言 → 持锚回应（锚由代码算出，不存在"复核改判"）
    if anchor and (is_challenge(user_input) or any(w in user_input for w in ASSERT_WORDS)):
        challenge = is_challenge(user_input)
        student_values = []
        if anchor.get("value") is not None and not any(w in user_input for w in NEGATION_WORDS):
            student_values = extract_numbers(user_input)
        assert_correct = (
            (not challenge)
            and bool(student_values)
            and any(numbers_close(v, anchor["value"]) for v in student_values)
        )
        if assert_correct:
            print(f"\n[系统判卷]✅ 你说的答案与标准答案一致（{anchor['answer']}）")
            scores.append(True)
            injections.append({
                "role": "system",
                "content": (
                    f"【判卷结果】学生主张的答案与标准答案【{anchor['answer']}】（程序验证）一致，判定正确。"
                    "请肯定学生，并简要给出推导作为确认，然后继续推进。"
                ),
            })
        else:
            print(f"\n[系统]✅ 本题答案【{anchor['answer']}】已由程序计算验证，老师将坚持。")
            injections.append({
                "role": "system",
                "content": (
                    f"【持锚回应】学生质疑或主张了不同答案。本题（{anchor['question']}）的标准答案是"
                    f"【{anchor['answer']}】，已经程序计算验证，不可更改。"
                    "请先确认学生说的是哪道题，然后一步步展示推导；若学生主张的数值有误，"
                    "温和指出他推导中出错的那一步（例如混淆了“直角边”与“直角边在斜边上的射影”）。"
                    "绝不改口。"
                ),
            })

    # ② 求助/求讲解 → 不判卷，换思路重讲
    elif anchor and not re.search(r"\d", user_input) and any(w in user_input for w in HELP_WORDS):
        print("\n[系统]你想听讲解，这道题先记着。")
        injections.append({
            "role": "system",
            "content": (
                f"【讲解请求】学生表示不会或想再听讲解。题目：{anchor['question']}，"
                f"标准答案：{anchor['answer']}（程序验证）。请把这一步讲清楚："
                "先列出已知条件，再写出所用公式，代入算出结果；讲完鼓励学生再试一次。"
            ),
        })

    # ③ 正常判卷
    elif anchor:
        passed, note = grade_answer(anchor, user_input)
        if passed == "skip":
            print("\n[系统]你切换了话题，刚才那道题先留着，想做了随时回来。")
        elif passed is not None:
            mark = "✅" if passed else "❌"
            print(f"\n[系统判卷]{mark} {note}")
            scores.append(passed)
            if not passed:
                save_wrong_question(anchor, user_input, note)
                question_text, answer_value = generate_variation()
                preset_anchor = make_anchor(question_text, answer_value)
                injections.append({
                    "role": "system",
                    "content": (
                        f"【判卷结果】学生回答错误：{note}。本题标准答案【{anchor['answer']}】（程序验证）。"
                        "请针对错误点讲解，讲完后把下面这道新题出给学生练习（切勿透露答案）：\n"
                        f"{question_text}\n（新题标准答案：{preset_anchor['answer']}，仅你可见）"
                    ),
                })
            else:
                injections.append({
                    "role": "system",
                    "content": (
                        f"【判卷结果】学生回答正确。标准答案【{anchor['answer']}】（程序验证）。"
                        "请肯定学生并继续推进，不要重复出原题。"
                    ),
                })

    # ④ 知识点卡注入（防"摄影定理"式编造）
    topic = match_knowledge(user_input)
    if topic:
        injections.append({
            "role": "system",
            "content": (
                f"【权威知识点】学生输入与「{topic}」高度相关（可能含错别字）。"
                f"讲解必须以下述表述为准，严禁编造或歪曲：\n{KNOWLEDGE[topic]}"
            ),
        })

    history.extend(injections)
    history.append({"role": "user", "content": user_input})
    try:
        reply = stream_reply(teacher["name"], history)
        history.append({"role": "assistant", "content": reply})
        # ⑤ 老师回复后锁定新题（题面重写去污染 + sympy 验证）
        if preset_anchor:
            anchor = preset_anchor
        else:
            anchor = lock_question(reply)
            if anchor:
                print("[系统]（新题标准答案已锁定并经程序验证）")
    except Exception as error:
        print("\n[网络错误，请检查：1.电脑是否联网 2.是否开了VPN/代理，然后重试]")
        print("详细信息：", type(error).__name__)

# ================= 5. 退出：保存聊天记录 =================
if len(history) > 1:
    filename = f"记录_{teacher['subject']}_{time.strftime('%m%d_%H%M')}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        for msg in history[1:]:
            if msg["role"] == "system":
                continue  # 内部指令不进聊天记录
            who = "学生" if msg["role"] == "user" else teacher["name"]
            f.write(f"{who}：{msg['content']}\n\n")
    if scores:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(f"\n[本次成绩：{scores.count(True)}/{len(scores)} 题正确]\n")
    print(f"\n本次辅导记录已保存到：{filename}")

if scores:
    correct_count = scores.count(True)
    print(f"📊 本次成绩：{correct_count}/{len(scores)} 题正确")
print("下次见！")