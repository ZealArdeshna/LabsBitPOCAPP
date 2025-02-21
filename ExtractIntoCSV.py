from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from document_loader import load_file
from pydantic import BaseModel, Field
from llm_model import LLMModel
from typing import List
import json
import csv
import re


class OneDevice(BaseModel):
    name: str = Field(description=" Name of the Fitness Device")
    features: List[str] = Field(description="List of Features, Include all features mentioned.")


class ListOfDevice(BaseModel):
    list_of_devices: List[OneDevice] = Field(
        description="List of dictionary with all Fitness Devices Name and it's Features.")


def extract_details_from_document():
    # Instantiate the parser with the new model.
    parser = PydanticOutputParser(pydantic_object=ListOfDevice)

    template = """

    You are good at extracting information from provided data. 
    Please review the provided document and provide a structure the following information into a JSON format:

    Name of the Fitness Device
    List of Features – Include all features mentioned, with proper formatting.

    Provide result in below format only:
    {format_instructions}

    Ensure that all extracted data is accurate, complete, and clearly formatted in JSON for easy integration.

    =================================================
    Document:
    {context}
    =================================================

    Json Format Result Only:
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    return prompt


def cleaning_json_result(response):
    match = re.search(r'```json\n(.*?)\n```', response.content, re.DOTALL)
    if match:
        llm_json_text = match.group(1)
        if isinstance(llm_json_text, str):
            json_data = json.loads(llm_json_text)
            return json_data


# Function to convert JSON data into CSV
def convert_to_csv(devices_data, filename='devices_data.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(['Device Name', 'Feature'])

        # Write each device's features to the CSV
        for device in devices_data['list_of_devices']:
            writer.writerow([device["name"], device["features"]])


if __name__ == "__main__":
    llm_model = LLMModel()

    loaded_document = load_file("data/unisport_fitness_2018.pdf")
    llm_chain = extract_details_from_document() | llm_model.azure_llm_model()
    str_result = llm_chain.invoke({"context": loaded_document})

    json_result = cleaning_json_result(str_result)
    print(json_result)
    convert_to_csv(json_result)
