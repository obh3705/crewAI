import json
import re
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


# 초성에 대한 숫자 매핑
def get_chosung(char):
    chosung_index = (ord(char) - 0xAC00) // 588
    return chosung_index % 19 + 1  # 1부터 19까지 순환

# 중성에 대한 숫자 매핑
def get_jungsung(char):
    jungsung_index = ((ord(char) - 0xAC00) % 588) // 28
    return jungsung_index % 21 + 1  # 1부터 21까지 순환

# 종성에 대한 숫자 매핑
def get_jongsung(char):
    jongsung_index = (ord(char) - 0xAC00) % 28
    return 0 if jongsung_index == 0 else (jongsung_index % 28) + 1

def is_korean(text):
    # 한글 유니코드 범위: 0xAC00(가)부터 0xD7A3(힣)까지
    return bool(re.match(r'^[\uac00-\ud7a3]+$', text))

def reduce_to_one_digit(num):
    while num >= 10:
        sum = 0
        while num > 0:
            sum += num % 10
            num //= 10
        num = sum
    return num

def analyze_korean_name(name):
    chosung_sum = 0
    jungsung_sum = 0
    jongsung_sum = 0

    for char in name:
        char_code = ord(char)
        chosung_index = (char_code - 0xAC00) // 588
        jungsung_index = ((char_code - 0xAC00) % 588) // 28
        jongsung_index = (char_code - 0xAC00) % 28

        chosung_sum += (chosung_index + 1)
        jungsung_sum += (jungsung_index + 1)
        jongsung_sum += jongsung_index if jongsung_index != 0 else 0

        # 자리수의 합을 계산하여 한 자리 숫자로 만들기
        chosung_sum = reduce_to_one_digit(chosung_sum)
        jungsung_sum = reduce_to_one_digit(jungsung_sum)
        jongsung_sum = reduce_to_one_digit(jongsung_sum)

    return {
        'chosung_sum': chosung_sum,
        'jungsung_sum': jungsung_sum,
        'jongsung_sum': jongsung_sum
    }

meanings = {
    'chosung': {
        1: ["독립성과 리더십.", "새로운 시작과 창조적인 에너지."],
        2: ["협력과 조화.", "친절하고 배려 깊은 성격."],
        3: ["소통과 표현.", "창의적이고 사교적인 특성."],
        4: ["안정성과 신뢰.", "질서와 책임감."],
        5: ["변화와 다양성.", "유연성과 모험심."],
        6: ["치유와 보살핌.", "가정과 사랑에 대한 중요성."],
        7: ["내면의 지혜와 직관.", "분석적이고 신비로운 면모."],
        8: ["권력과 성공.", "경제적 안정과 효율성."],
        9: ["인도와 이타심.", "보편적 사랑과 인류애."]
    },
    'jungsung': {
        1: ["자기주도적 감정.", "독립적인 감정 상태."],
        2: ["조화와 균형을 추구하는 감정.", "타인과의 관계에서 강한 유대감."],
        3: ["창의력과 표현력이 풍부한 감정.", "유쾌함과 낙관주의."],
        4: ["안정과 질서를 선호하는 내면의 상태.", "신뢰성과 규율."],
        5: ["변화에 대한 열린 감정.", "새로운 경험에 대한 열망."],
        6: ["사랑과 보호를 중시하는 감정.", "가족과 친구에 대한 헌신."],
        7: ["심오한 사색과 성찰.", "내면의 진리를 찾는 경향."],
        8: ["물질적, 경제적 안정에 대한 욕구.", "실용적인 감정 처리."],
        9: ["인류애와 관용.", "세상에 대한 깊은 이해와 공감."]
    },
    'jongsung': {
        0: ["잠재적 무한의 가능성."],
        1: ["성취된 목표의 시작과 완성.", "독립적인 행동의 결말."],
        2: ["협동과 파트너십의 완성.", "관계에서의 조화."],
        3: ["소통과 창의력의 결실.", "사회적 인정의 획득."],
        4: ["계획과 프로젝트의 안정적 종결.", "실용적 성과의 완성."],
        5: ["변화와 자유의 추구에서 오는 결과.", "적응력의 표현."],
        6: ["가정과 커뮤니티에 대한 헌신의 결실.", "이타적 행동의 완성."],
        7: ["지식과 지혜의 추구에서 얻은 결론.", "영적 발견."],
        8: ["물질적, 직업적 성공의 최종 단계.", "경제적 자립."],
        9: ["인도적 노력의 완성.", "세상에 대한 긍정적인 영향."]
    }
}

def display_meaning(number, type):
    type_name = ''
    if type == 'chosung':
        type_name = '초성'
    elif type == 'jungsung':
        type_name = '중성'
    elif type == 'jongsung':
        type_name = '종성'

    st.write(f"### {type_name}의 숫자 {number}")
    st.write(meanings[type][number][0])
    st.write(meanings[type][number][1])

class NumberologyAgent:
    # gpt-4o-mini 모델을 사용하여 이름을 분석하는 에이전트 result를 보고 에이전트의 성격을 생성
    def __init__(self, name):
        self.name = name
        self.result = analyze_korean_name(name)
        self.prompt = PromptTemplate.from_template("""
                                                   당신은 이름점술가입니다. 이름의 결과를 보고 현실적인 성격을 분석합니다. 결과: {result} 
                                                   성격 분석은 다음과 같은 형식으로 작성됩니다:
                                                   # 이름의 의미

                                                   # 종합 분석
                                                   """)
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.parser = StrOutputParser()
        self.chain = self.prompt | self.llm | self.parser

    def action(self):
        return self.chain.invoke({"result":self.result})

# Streamlit UI
st.title("넘버로지 에이전트")

name = st.text_input("이름을 입력하세요:")
submit = st.button("에이전트 생성")

if name or submit:
    if is_korean(name):
        result = analyze_korean_name(name)
        agent = NumberologyAgent(name)
        # display_meaning(result['chosung_sum'], 'chosung')
        # display_meaning(result['jungsung_sum'], 'jungsung')
        # display_meaning(result['jongsung_sum'], 'jongsung')

        action = agent.action()
        # 종합 분석을 변수에 저장
        analysis = action.split("# 종합 분석")[1]
        st.write(action)

        # 에이전트의 이름과 성격을 json으로 저장
        with open('./agent.json', 'w') as f:
            json.dump({"name": name, "analysis": analysis}, f)

    else:
        st.write("Please enter a valid Korean name.")