from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task, llm
from langchain_openai import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from crewai_tools import YoutubeChannelSearchTool, YoutubeVideoSearchTool
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool

# Uncomment the following line to use an example of a custom tool
# from mas.tools.custom_tool import MyCustomTool

# Check our tools documentations for more information on how to use them
# from crewai_tools import SerperDevTool

@CrewBase
class MasCrew():
	"""Mas crew"""
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	@llm
	def my_llm(self):
		return ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

	@agent
	def researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['researcher'],
			# tools=[DuckDuckGoSearchRun(name="Search")],
			tools=[YahooFinanceNewsTool(top_k=5)],
			verbose=True
		)

	@agent
	def reporting_analyst(self) -> Agent:
		return Agent(
			config=self.agents_config['reporting_analyst'],
			verbose=True
		)
	
	@agent
	def translator(self) -> Agent:
		return Agent(
			config=self.agents_config['translator'],
			verbose=True
		)

	@task
	def research_task(self) -> Task:
		return Task(
			config=self.tasks_config['research_task'],
		)

	@task
	def reporting_task(self) -> Task:
		return Task(
			config=self.tasks_config['reporting_task'],
			output_file='report.md'
		)
	
	@task
	def translate_task(self) -> Task:
		return Task(
			config=self.tasks_config['translate_task'],
			output_file='report.md'
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the Mas crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)