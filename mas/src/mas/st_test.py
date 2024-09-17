import streamlit as st
from langchain.agents import AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.schema import AgentFinish
import re

from dotenv import load_dotenv
import os

# API KEY 정보 로드
load_dotenv()

# LLM 설정
llm = OpenAI(temperature=0)

# 커스텀 출력 파서 정의
class CustomOutputParser(AgentOutputParser):
    def parse(self, text):
        # 디버깅을 위해 LLM의 출력을 출력합니다.
        # Streamlit에서는 st.write를 사용하여 출력할 수 있습니다.
        st.write("LLM 출력:", text)
        # '답변:'으로 시작하는 부분을 추출합니다.
        match = re.search(r"답변:(.*)", text, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            return AgentFinish(return_values={"output": answer}, log=text)
        else:
            # 매칭되지 않을 경우 원본 텍스트 반환
            return AgentFinish(return_values={"output": text.strip()}, log=text)

# 에이전트 1: 보수적인 성향
def run_conservative_agent(question):
    try:
        conservative_prompt = PromptTemplate(
            input_variables=["input"],
            template="""
당신은 신중하고 분석적인 전문가입니다. 위험을 최소화하고 검증된 방법을 선호합니다.

질문: {input}
답변을 '답변:'으로 시작하여 작성하세요.
답변:"""
        )

        # LLM 체인 생성
        llm_chain = LLMChain(llm=llm, prompt=conservative_prompt)

        # 출력 파서 정의
        output_parser = CustomOutputParser()

        # 중단 시퀀스 정의
        stop = ["\n질문:", "\n답변:"]

        # 에이전트 초기화
        agent_conservative = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=stop
        )

        agent_executor_conservative = AgentExecutor.from_agent_and_tools(
            agent=agent_conservative,
            tools=[],
            verbose=False
        )

        response = agent_executor_conservative.run(question)
        return response

    except Exception as e:
        st.error(f"보수적인 에이전트에서 오류 발생: {e}")
        return "에이전트가 응답하지 않았습니다."

# 에이전트 2: 혁신적인 성향
def run_innovative_agent(question):
    try:
        innovative_prompt = PromptTemplate(
            input_variables=["input"],
            template="""
당신은 창의적이고 혁신적인 전문가입니다. 새로운 아이디어와 접근 방식을 적극적으로 제안합니다.

질문: {input}
답변을 '답변:'으로 시작하여 작성하세요.
답변:"""
        )

        # LLM 체인 생성
        llm_chain = LLMChain(llm=llm, prompt=innovative_prompt)

        # 출력 파서 정의
        output_parser = CustomOutputParser()

        # 중단 시퀀스 정의
        stop = ["\n질문:", "\n답변:"]

        # 에이전트 초기화
        agent_innovative = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=stop
        )

        agent_executor_innovative = AgentExecutor.from_agent_and_tools(
            agent=agent_innovative,
            tools=[],
            verbose=False
        )

        response = agent_executor_innovative.run(question)
        return response

    except Exception as e:
        st.error(f"혁신적인 에이전트에서 오류 발생: {e}")
        return "에이전트가 응답하지 않았습니다."

# 에이전트 회의 함수
def agent_discussion(response1, response2):
    try:
        discussion_prompt = PromptTemplate(
            input_variables=["agent1_response", "agent2_response"],
            template="""
두 전문가가 다음과 같은 의견을 제시했습니다:

에이전트 1: "{agent1_response}"

에이전트 2: "{agent2_response}"

두 에이전트는 서로의 의견을 검토하고 토론합니다. 회의 내용을 요약하고, 최종 합의된 전략을 제시하세요.
전략:"""
        )

        # LLM 체인 생성
        llm_chain = LLMChain(llm=llm, prompt=discussion_prompt)

        # 출력 파서 정의
        class DiscussionOutputParser(AgentOutputParser):
            def parse(self, text):
                st.write("LLM 출력 (회의):", text)
                match = re.search(r"전략:(.*)", text, re.DOTALL)
                if match:
                    strategy = match.group(1).strip()
                    return AgentFinish(return_values={"output": strategy}, log=text)
                else:
                    return AgentFinish(return_values={"output": text.strip()}, log=text)

        output_parser = DiscussionOutputParser()

        # 중단 시퀀스 정의
        stop = ["\n전략:", "\n에이전트"]

        # 에이전트 초기화
        discussion_agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=stop
        )

        discussion_executor = AgentExecutor.from_agent_and_tools(
            agent=discussion_agent,
            tools=[],
            verbose=False
        )

        final_result = discussion_executor.run({
            "agent1_response": response1,
            "agent2_response": response2
        })

        return final_result

    except Exception as e:
        st.error(f"회의 에이전트에서 오류 발생: {e}")
        return "회의 결과를 생성하지 못했습니다."

# Streamlit 앱 구성
st.title('멀티 에이전트 시스템 데모')
st.write('질문을 입력하고 에이전트들의 답변과 회의 결과를 확인하세요.')

# 사용자 입력 받기
user_input = st.text_input('질문을 입력하세요:')

if st.button('에이전트 실행'):
    with st.spinner('에이전트들이 열심히 고민 중입니다...'):
        # 에이전트 실행
        response_conservative = run_conservative_agent(user_input)
        response_innovative = run_innovative_agent(user_input)
        final_result = agent_discussion(response_conservative, response_innovative)

    # 결과 표시
    st.subheader('에이전트들의 답변')
    st.write('**에이전트 1 (보수적인 성향):**')
    st.write(response_conservative)
    st.write('**에이전트 2 (혁신적인 성향):**')
    st.write(response_innovative)

    st.subheader('회의 결과')
    st.write(final_result)

    # 에이전트 대화 표시
    st.subheader('에이전트 대화')
    st.markdown('**에이전트 1:**')
    st.info(response_conservative)
    st.markdown('**에이전트 2:**')
    st.success(response_innovative)
    st.markdown('**회의 결과:**')
    st.warning(final_result)