import streamlit as st
import operator
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
    SystemMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph, START
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import functools
from typing import Sequence
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

memory = MemorySaver()
config = {"configurable": {"thread_id": "test-thread"}, "recursion_limit": 150}

# Define the agent state
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    sender: str
    pro_position: str
    con_position: str
    full_text: Sequence[str]

class DiscussionAgent():
    PROMPT_TEMPLATE: str = """
    당신은 최고의 토론자입니다. {topic}에 대해 설득력있는 주장을 제시하십시오.
    {system_message}
    """
    
    def __init__(self, llm, name, topic, agree="찬성"):
        self.llm = llm
        self.name = name
        self.topic = topic
        self.agree = agree


    def argument_action(self, state, agent, name):
        
        action_prompt = """ 당신은 주제에 대해 {self.agree} 입장을 표명해야합니다. 구조를 사용하지 않고 실제 토론에서 말하듯이 말하십시오.
        1.	명확한 주장 설정: 자신의 주장을 간결하고 명확하게 표현합니다. 청중이 한 번에 이해할 수 있을 정도로 간단하게 핵심 메시지를 전달하는 것이 중요합니다.
        2.	체계적 구조 사용 (PREP 방식):
            •	Point (주장): 먼저 자신이 주장하는 핵심 내용을 명확히 진술합니다.
            •	Reason (이유): 주장을 뒷받침하는 이유를 제시합니다. 이때 논리적으로 타당한 이유가 필요합니다.
            •	Example (예시): 구체적인 사례나 데이터를 사용해 주장의 신뢰성을 높입니다. 현실적이고 실질적인 예시는 청중의 이해를 도와줍니다.
            •	Point (재확인): 마지막으로 주장과 이유를 다시 한 번 요약해 강조합니다.
        3.	근거의 신뢰성 확보: 자신의 주장을 뒷받침할 수 있는 통계, 연구 결과, 전문가의 의견 등의 객관적인 자료를 사용해 설득력을 높입니다. 다양한 출처에서 얻은 신뢰할 만한 정보가 있을수록 좋습니다.
        4.	논리적 연결: 각 주장과 이유, 근거들이 논리적으로 잘 연결되도록 합니다. 논리적 흐름이 단절되지 않게 주의하고, 각 부분이 자연스럽게 이어질 수 있도록 신경 씁니다.
        5.	청중 고려: 청중이 누구인지, 그들의 관심사나 이해 수준을 고려하여 주장을 구성하는 것이 중요합니다. 청중이 이해할 수 있는 언어와 방식으로 표현하고, 그들이 왜 자신의 주장을 받아들여야 하는지 명확히 설명합니다.
        6.	감정적 요소 활용: 감정적인 요소도 중요한 설득의 도구입니다. 청중의 감정에 호소할 수 있는 이야기를 적절히 사용하면 주장에 대한 공감대를 형성할 수 있습니다. 다만, 지나친 감정적인 호소는 오히려 역효과를 불러올 수 있으므로 균형을 잘 맞춰야 합니다.
        7.	반대 의견 고려: 자신의 주장에 반대할 수 있는 의견을 미리 예측하고, 이에 대한 반박을 준비합니다. 이를 통해 자신의 주장이 더 탄탄해지고, 상대방의 비판을 미리 잠재울 수 있습니다.
        """

        action_prompt = action_prompt.format(self=self)
        prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                self.PROMPT_TEMPLATE,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
        )
        prompt = prompt.partial(system_message=action_prompt, topic=self.topic)
        return prompt

    def rebuttal_action(self, state, agent, name):
        action_prompt = """당신은 앞 토론자의 의견에 대한 반론을 작성하십시오.
        반론은 다음과 같습니다. 구조를 사용하지 않고 실제 토론에서 말하듯이 말하시오.
        1.	상대방 주장 분석: 상대방의 주장이나 증거를 이해하고, 그 내용의 강점과 약점을 분석합니다.
        2.	논리적 허점 발견: 상대방의 주장 중 논리적으로 일관성이 없거나 모순이 있는 부분을 찾습니다.
        3.	반박 근거 제시: 상대방의 주장을 반박할 수 있는 구체적인 증거나 논리적인 이유를 제시합니다. 이때 믿을 만한 자료나 사실을 사용하면 반박이 더욱 설득력 있게 됩니다.
        4.	대안 제시: 반론 이후에는 대안적인 관점이나 새로운 방향을 제시함으로써 자신의 입장을 강화합니다.
        """
        prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                self.PROMPT_TEMPLATE,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
        )
        prompt = prompt.partial(system_message=action_prompt, topic=self.topic)
        return prompt
    
    def agent_node(self, state, agent, name, action):
        # if state[pro_position] == "찬성":
        if self.agree == "찬성":
            self.agree = state["pro_position"]
        else:
            self.agree = state["con_position"]

        if action == "argument":
            prompt = self.argument_action(state, agent, name)
        elif action == "rebuttal":
            prompt = self.rebuttal_action(state, agent, name)
        else:
            raise ValueError("Invalid action")
        
        agent = prompt | self.llm
        
        result = agent.invoke(state)
        
        if 'full_text' not in state:
            state['full_text'] = []
        state['full_text'].append(result.content + "\n")

        if isinstance(result, ToolMessage):
            pass
        else:
            result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
            
        if action == "argument":
            return {
                "messages": [result],
                # "full_text": state['full_text'],
                "sender": name,
            }
        elif action == "rebuttal":
            return {
                "messages": [result],
                # "full_text": state['full_text'],
                "sender": name,
            }
        else:
            return {
                "messages": [result],
                # "full_text": state['full_text'],
                "sender": name,
            }
    
class RecordAgent():
    PROMPT_TEMPLATE: str = """
    당신은 회의록 작성 AI입니다.
    토론 내용을 정리하여 보고서로 작성하십시오.
    형식은 markdown 형식을 따라야합니다.
    토론의 회의록은 다음과 같은 형식으로 작성됩니다:
    1.	토론 정보:
	•	토론 제목: 예를 들어, “온라인 수업의 장점과 단점에 대한 찬반 토론”.
	2.	목적 및 주요 안건:
	•	토론의 목적을 간략히 설명합니다(예: 온라인 수업의 효율성에 대한 찬반 검토).
	•	주요 안건: 찬성 측과 반대 측의 논거들을 논의.
	3.	찬성 측 의견 요약:
	•	주요 주장: 온라인 수업이 더 효율적이라는 주장.
	•	근거: 시간과 장소의 제약이 없고, 학생들에게 더 많은 자율성을 부여할 수 있다는 점.
	•	구체적 사례: 특정 학교의 사례에서 온라인 수업이 시험 성적 향상에 기여했다는 데이터.
	4.	반대 측 의견 요약:
	•	주요 주장: 온라인 수업이 비효율적일 수 있다는 주장.
	•	근거: 학생들의 집중도가 낮아지고, 상호작용이 줄어든다는 점.
	•	구체적 사례: 많은 학생들이 온라인 환경에서 학습 의욕 저하를 겪고 있다는 설문조사 결과.
	5.	토론 중간 논의 내용:
	•	양측이 질문하고 반박한 내용을 간단히 요약합니다.
	•	찬성 측이 제기한 질문에 대한 반대 측의 응답과 그에 대한 반론이 어떻게 이어졌는지 기록합니다.
    6.  청중의 의견:
    •   청중의 의견을 요약하고, 찬성과 반대 중 더 설득력 있는 쪽을 선택합니다.
	7.	결론 및 합의된 사항:
	•	결론: 최종적으로 양측이 합의한 사항이 있다면 명시합니다. 합의가 이루어지지 않았더라도 각 측의 주장을 정리해 요약합니다.
	•	중재자 의견: 중재자가 있다면, 중립적으로 요약한 의견이나 결론을 기록합니다.
	8.	추후 조치 사항:
	•	토론의 결과에 따라 추가로 논의가 필요한 사항이나 앞으로 진행해야 할 활동들을 기록합니다.
    """
    
    def __init__(self, llm, name):
        self.llm = llm
        self.name = name

    def agent_node(self, state, agent, name):
        # action_prompt = self.PROMPT_TEMPLATE
        prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                self.PROMPT_TEMPLATE,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
        )
        # prompt = prompt.partial()
        agent = prompt | self.llm
        
        result = agent.invoke(state)

        if isinstance(result, ToolMessage):
            pass
        else:
            result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)

        # Markdown 파일에 내용 작성
        md_file_path = "langchain_response.md"

        with open(md_file_path, "w") as file:
            file.write(result.content)

        print(f"Markdown file saved successfully at {md_file_path}")

        return {
            "messages": [result],
            "sender": name,
        }
    
class ModeratorAgent():
    PROMPT_TEMPLATE: str = """
    당신은 토론의 사회자입니다. 
    {topic}을 보고 찬성측 주제를 제시하고, 반대측 주제를 제시하십시오.
    입장의 설명을 하지 않아야합니다.
    답변은 다음과 같은 형식을 따라야합니다:
    1. 찬성 주제: [찬성 주제]
    2. 반대 주제: [반대 주제]
    """
    
    def __init__(self, llm, name, topic):
        self.llm = llm
        self.name = name
        self.topic = topic

    def agent_node(self, state, agent, name):
        # action_prompt = self.PROMPT_TEMPLATE
        prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                self.PROMPT_TEMPLATE,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
        )
        prompt = prompt.partial(topic = self.topic)
        agent = prompt | self.llm
        
        result = agent.invoke(state)

        # 찬성 및 반대 입장을 파싱합니다.
        pro_position = result.content.split("찬성 주제: ")[1].split("\n")[0].strip()
        con_position = result.content.split("반대 주제: ")[1].strip()

        if isinstance(result, ToolMessage):
            pass
        else:
            result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
        return {
            "messages": [result],
            "sender": name,
            "pro_position": pro_position,
            "con_position": con_position,
        }

class AudienceAgent():
    PROMPT_TEMPLATE: str = """
    당신은 청중입니다. 토론을 듣고 의견을 말해주세요.
    마지막은 찬성과 반대 입장 중 더 설득력 있는 쪽을 선택해주세요.
    결과는 아래 형식을 따르지 않고 토론에서 청중이 말하듯이 작성하십시오.
    1.	논리적 일관성 평가:
	•	각 주장이 논리적으로 일관성이 있는지 살펴봅니다. 주장과 근거가 잘 연결되어 있으며, 결론으로 자연스럽게 이어지는지 판단합니다.
	•	토론 중 모순되거나 사실과 일치하지 않는 내용이 있다면 그것을 확인하고 점수에 반영합니다.
	2.	근거의 질과 신뢰성:
	•	각 주장을 뒷받침하는 근거가 신뢰할 만한 자료나 데이터를 기반으로 하는지 확인합니다. 구체적이고 객관적인 증거를 사용하는지, 근거가 논리적으로 주장과 연관성이 있는지를 평가합니다.
	•	주관적인 의견이나 검증되지 않은 주장은 설득력이 떨어질 수 있음을 인지합니다.
	3.	상대방 주장에 대한 반박의 강도:
	•	찬성 또는 반대 측이 상대방의 주장에 대해 어떻게 반박하는지를 평가합니다. 상대방 주장의 허점을 논리적이고 설득력 있게 지적하고 근거로 반박하는지, 아니면 단순히 반대만 하고 있는지를 살펴봅니다.
	•	반박 과정에서 새로운 근거를 제시하거나 상대방 논리를 무너뜨리는 방식이 얼마나 강력했는지 평가합니다.
	4.	감정적 호소와 논리적 설득의 균형:
	•	논리적 설득이 강한지, 혹은 감정에 호소하는 부분이 지나치지 않은지 평가합니다. 감정적 호소도 중요한 요소이지만, 논리적 타당성을 넘어서 설득력을 얻기 위한 수단으로 사용되었는지 살펴봅니다.
	•	너무 감정적이거나 극단적인 주장보다는 논리적이고 근거 있는 주장을 우선적으로 평가합니다.
	5.	명확성과 이해도:
	•	각 측의 주장이 얼마나 명확하게 전달되었는지를 평가합니다. 복잡한 개념을 간단하고 이해하기 쉽게 설명했는지, 청중이 쉽게 따라올 수 있도록 했는지를 판단합니다.
	•	주장과 근거를 명료하게 정리한 쪽이 더 설득력 있게 느껴질 가능성이 높습니다.
	6.	균형 잡힌 시각:
	•	찬성과 반대 측이 모두 자신의 주장을 과장하지 않고 균형 잡힌 시각으로 논의하는지 평가합니다. 상대방의 주장에 대한 인정이나 객관적인 접근을 보이는 경우가 있다면, 이는 논의의 신뢰성을 높여줍니다.
	7.	질문과 응답의 품질:
	•	질문이 주장을 명확히 하고 토론을 깊게 만들었는지, 혹은 상대방의 논리적 결함을 지적했는지를 살펴봅니다.
	•	상대방의 질문에 대한 응답이 논리적이고 일관성 있게 이루어졌는지도 중요한 평가 요소입니다.
	8.	결과 도출:
	•	찬성과 반대의 각 측이 제시한 주장을 정리하고, 앞에서 언급한 논리성, 신뢰성, 반박의 강도 등을 바탕으로 자신만의 평가 기준을 사용해 최종적으로 어느 쪽이 더 설득력 있었는지를 판단합니다.
    """
    
    def __init__(self, llm, name):
        self.llm = llm
        self.name = name

    def agent_node(self, state, agent, name):
        # action_prompt = self.PROMPT_TEMPLATE
        prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                self.PROMPT_TEMPLATE,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
        )
        # prompt = prompt.partial()
        agent = prompt | self.llm
        
        result = agent.invoke(state)

        if isinstance(result, ToolMessage):
            pass
        else:
            result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
        return {
            "messages": [result],
            # "full_text": state['full_text'],
            "sender": name,
        }

def run_meeting(topic: str):
    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4o-mini")
    llm2 = ChatAnthropic(model="claude-3-haiku-20240307")

    moderator = ModeratorAgent(llm, "moderator", topic)
    moderator_node = functools.partial(moderator.agent_node, agent=moderator, name="moderator")

    agent1 = DiscussionAgent(llm, "agent1", topic, agree="찬성")
    agent1_node = functools.partial(agent1.agent_node, agent=agent1, name="agent1", action="argument")
    agent1_rebuttal = functools.partial(agent1.agent_node, agent=agent1, name="agent1", action="rebuttal")

    agent2 = DiscussionAgent(llm, "agent2", topic, agree="반대")
    agent2_node = functools.partial(agent2.agent_node, agent=agent2, name="agent2", action="argument")
    agent2_rebuttal = functools.partial(agent2.agent_node, agent=agent2, name="agent2", action="rebuttal")

    audience = AudienceAgent(llm, "audience")
    audience_node = functools.partial(audience.agent_node, agent=audience, name="audience")

    agent4 = RecordAgent(llm, "agent4")
    agent4_node = functools.partial(agent4.agent_node, agent=agent4, name="agent4")

    # StateGraph configuration
    workflow = StateGraph(AgentState)
    workflow.add_node("moderator", moderator_node)
    workflow.add_node("agent1", agent1_node)
    workflow.add_node("agent2", agent2_node)
    workflow.add_node("agent1_rebuttal", agent1_rebuttal)
    workflow.add_node("agent2_rebuttal", agent2_rebuttal)
    workflow.add_node("audience", audience_node)
    workflow.add_node("agent4", agent4_node)

    # Define the edges between the nodes
    workflow.add_edge(START, "moderator")
    workflow.add_edge("moderator", "agent1")
    workflow.add_edge("agent1", "agent2_rebuttal")
    workflow.add_edge("agent2_rebuttal", "agent2")
    workflow.add_edge("agent2", "agent1_rebuttal")
    workflow.add_edge("agent1_rebuttal", "audience")
    workflow.add_edge("audience", "agent4")
    workflow.add_edge("agent4", END)

    # Compile the workflow
    graph = workflow.compile(checkpointer=memory)

    # Start the discussion with the given topic
    events = graph.stream(
        {
            "messages": [
                HumanMessage(
                    content=topic,
                )
            ],
        },
        config,
    )

    # Display results in a readable format
    for s in events:
        try:
            if isinstance(s, dict):
                for agent_name, agent_data in s.items():
                    if 'messages' in agent_data:
                        sender = agent_data['sender']
                        message_content = agent_data['messages'][0] if agent_data['messages'] else "No content"
                        
                        if hasattr(message_content, 'content'):
                            message_content = message_content.content

                        st.write(f"**{sender}의 응답:**")
                        st.write(message_content.replace("\\n", "\n"))
                        st.write("---")
        except Exception as e:
            st.error(f"Error processing event: {e}")

# 이 함수를 다른 코드에서 호출하여 회의를 실행할 수 있습니다.
# 예시: run_meeting("한국의 의대 증원 문제에 대해 토론해보자.")