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
    full_text: Sequence[str]


# Define the agent
class Agent:
    PROMPT_TEMPLATE: str = """
    당신은 최고의 {topic} 전문가입니다.
    {system_message}
    """

    def __init__(self, llm, name, topic, propensity="긍정적"):
        self.llm = llm
        self.name = name
        self.topic = topic
        self.propensity = propensity

    def action(self):
        
        action_prompt = """ 당신은 {self.topic}에 대해 {self.propensity}한 성격을 가지고 있습니다.
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
    
    
    def agent_node(self, state, agent, name):
        prompt = self.action()
        
        agent = prompt | self.llm
        
        result = agent.invoke(state)

        if 'full_text' not in state:
            state['full_text'] = []
        state['full_text'].append(result.content + "\n")

        if isinstance(result, ToolMessage):
            pass
        else:
            result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
            
        return {
            "messages": [result],
            # "full_text": state['full_text'],
            "sender": name,
        }
    
class FeedbackAgent:
    PROMPT_TEMPLATE: str = """
    당신은 최고의 {topic} 전문가입니다.
    {system_message}
    """

    def __init__(self, llm, name, topic):
        self.llm = llm
        self.name = name
        self.topic = topic

    def action(self):
        
        action_prompt = """ 당신은 {self.topic}에 대해 중립적인 입장에서 피드백을 제공합니다.
        앞선 의견을 듣고 부족한 부분이 있는지 화인하고 피드백을 제공해주세요.
        피드백 내용은 답변이 한 쪽으로 치우치지 않도록 주의해주세요.
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
    
    def agent_node(self, state, agent, name):
        prompt = self.action()
        
        agent = prompt | self.llm
        
        result = agent.invoke(state)

        if 'full_text' not in state:
            state['full_text'] = []
        state['full_text'].append(result.content + "\n")

        if isinstance(result, ToolMessage):
            pass
        else:
            result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
            
        return {
            "messages": [result],
            # "full_text": state['full_text'],
            "sender": name,
        }

class JudgementAgent:
    PROMPT_TEMPLATE: str = """
    당신은 최고의 {topic} 전문가입니다.
    {system_message}
    """

    def __init__(self, llm, name, topic):
        self.llm = llm
        self.name = name
        self.topic = topic

    def action(self):
        
        action_prompt = """ 당신은 {self.topic}에 대해 중립적인 의견을 가지고 있습니다.
        대화를 보고 판단을 내려 더 효율적인 의사결정을 도와주세요.
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
    
    def agent_node(self, state, agent, name):
        prompt = self.action()
        
        agent = prompt | self.llm
        
        result = agent.invoke(state)

        if isinstance(result, ToolMessage):
            pass
        else:
            result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
            
        return {
            "messages": [result],
            "sender": name,
        }
    

def run_meeting(topic: str):
    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4o-mini")
    llm2 = ChatAnthropic(model="claude-3-haiku-20240307")

    agent1 = Agent(llm, "agent1", topic, propensity="긍정적")
    agent1_node = functools.partial(agent1.agent_node, agent=agent1, name="agent1")

    agent2 = FeedbackAgent(llm, "agent2", topic)
    agent2_node = functools.partial(agent2.agent_node, agent=agent2, name="agent2")

    agent3 = JudgementAgent(llm, "agent3", topic)
    agent3_node = functools.partial(agent3.agent_node, agent=agent3, name="agent3")

    # StateGraph configuration
    workflow = StateGraph(AgentState)
    workflow.add_node("agent1", agent1_node)
    workflow.add_node("agent2", agent2_node)
    workflow.add_node("agent3", agent3_node)

    # Define the edges between the nodes
    workflow.add_edge(START, "agent1")
    workflow.add_edge("agent1", "agent2")
    workflow.add_edge("agent2", "agent3")
    workflow.add_edge("agent3", END)

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