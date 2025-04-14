import re
import webbrowser
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks import get_openai_callback

class SpecificAgent:
    def __init__(self, prompt_template: str, api_key: str, model: str = "gpt-4o",
                 output_processing=None, provider: str = "openai", name: str = "Super Agent"):
        """
        Initializes the SpecificAgent with a prompt template and LLM configuration.
        Supports 'openai' and 'gemini' providers using LangChain.
        """
        self.prompt_template = prompt_template
        self.api_key = api_key
        self.model = model
        self.provider = provider.lower()
        self.output_processing = output_processing
        self.name = name

        if self.provider == "openai":
            self.llm = ChatOpenAI(model=model, api_key=api_key)
        elif self.provider == "gemini":
            self.llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def process_output(self, response: str):
        """
        Processes the response based on the specified output processing type.
        """
        if self.output_processing == "CSV2GRAPH":
            code_match = re.search(r"```python(.*?)```", response, re.DOTALL)
            return code_match.group(1).strip() if code_match else response

        elif self.output_processing == "QUESTIONS":
            code_match = re.search(r"```python(.*?)```", response, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
                local_vars = {}
                exec(code, {}, local_vars)
                return local_vars.get('questions')
            return None

        elif self.output_processing == "SCHEMA_EXTRACT":
            print(response)
            schema_match = re.search(r"###START_SCHEMA(.*?)###END_SCHEMA", response, re.DOTALL)
            if schema_match:
                schema_code = schema_match.group(1).strip()
                local_vars = {}
                exec(schema_code, {}, local_vars)
                schema = local_vars.get('schema')
                if schema:
                    rel_props = set()
                    for rel in schema.get('relationships', []):
                        rel_props.update(rel.get('properties', []))

                    for entity_name, entity_data in schema.get('entities', {}).items():
                        entity_constraints = set(entity_data.get('constraints', []))
                        current_attributes = set(entity_data.get('attributes', []))
                        missing_constraints = entity_constraints - current_attributes
                        if missing_constraints:
                            entity_data['attributes'] = list(current_attributes.union(missing_constraints))
                        indexes = set(entity_data.get('indexes', []))
                        to_remove = (rel_props & current_attributes) - indexes - entity_constraints
                        entity_data['attributes'] = [
                            attr for attr in entity_data['attributes'] if attr not in to_remove
                        ]
                    return schema
            return None

        elif self.output_processing == "HTML_REPORT":
            html_match = re.search(r"```html(.*?)```", response.replace('\n', ''), re.DOTALL)
            if html_match:
                html_content = html_match.group(1).strip()
                with open('report.html', 'w', encoding='utf-8') as file:
                    file.write(html_content)
                webbrowser.open('report.html')
            else:
                print("No HTML content found in the message.")

        return response

    def invoke(self, **data) -> str:
        """
        Invokes the agent with the configured prompt and data.
        Uses LangChain's callback to print cost, input tokens, and output tokens.
        """
        formatted_prompt = self.prompt_template.format(**data) if data else self.prompt_template
    
        if self.provider == "openai":
            with get_openai_callback() as cb:
                response = self.llm.invoke(formatted_prompt)
                print(f"[Agent Name] {self.name}")
                print(f"[Token Usage] Input: {cb.prompt_tokens}, Output: {cb.completion_tokens}, Total: {cb.total_tokens}")
                print(f"[Cost] ${cb.total_cost:.6f}")
        elif self.provider == "gemini":
            with get_openai_callback() as cb:
                response = self.llm.invoke(formatted_prompt)
                print(f"[Agent Name] {self.name}")
                print(f"[Token Usage] Input: {cb.prompt_tokens}, Output: {cb.completion_tokens}, Total: {cb.total_tokens}")
                print(f"[Cost] ${cb.total_cost:.6f}")
    
        return self.process_output(response.content) if self.output_processing else response.content

    @classmethod
    def create_agent(cls, prompt_template: str, api_key: str, model: str = "gpt-4o",
                     output_processing=None, provider: str = "openai", name: str = "Super Agent"):
        """
        Factory method to create a new instance of the agent.
        """
        return cls(prompt_template, api_key, model, output_processing, provider, name)