import logging
import os
import sys
from rasa.shared.utils.io import read_json_file, write_yaml
from typing import Any, Dict, List, Text, Optional, Union
from pathlib import Path
from rasa.shared.nlu.constants import INTENT, ENTITIES, TEXT
from rasa.shared.nlu.training_data.util import transform_entity_synonyms
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.formats.rasa_yaml import RasaYAMLWriter
from rasa.shared.nlu.training_data.formats.readerwriter import (
    TrainingDataReader,
    TrainingDataWriter,
)
from rasa.utils.converter import TrainingDataConverter
from ruamel.yaml import StringIO
from collections import OrderedDict

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

KEY_DOMAIN = "domain"
KEY_INTENTS = "intents"
KEY_ENTITIES = "entities"
KEY_RESPONSES = "responses"

class RasaYAMLWriterDomain(RasaYAMLWriter):
    """Extends RasaYAMLWriter to also create domain file"""
    from rasa.shared.utils.io import write_yaml

    def __init__(self):
        super().__init__()

    @staticmethod
    def process_domain_examples_by_key(
        training_examples: Dict[Text, List[Union[Dict, Text]]],
        key_name: Text,
    ) -> List[OrderedDict]:
        """Prepares training examples  to be written to YAML.

        This can be any NLU training data (intent examples, lookup tables, etc.)

        Args:
            training_examples: Multiple training examples. Mappings in case additional
                values were specified for an example (e.g. metadata) or just the plain
                value.
            key_name: The top level key which the examples belong to (e.g. `intents`)

        Returns:
            NLU training data examples prepared for writing to YAML.
        """
        intents = []

        # for intent_name in training_examples.items():
        #     intent: OrderedDict[Text, Any] = OrderedDict()
        #     intent[key_name] = intent_name
        #     intents.append(intent)
        intents = ["intent1", "intent2"]

        return intents

    @classmethod
    def process_domain_intents(cls, training_data: "TrainingData") -> List[OrderedDict]:
        """Serializes the intents."""
        return RasaYAMLWriterDomain.process_domain_examples_by_key(
            cls.prepare_training_examples(training_data),
            KEY_INTENTS,
        )

    @classmethod
    def domain_data_to_dict(
        cls, training_data: "TrainingData"
    ) -> Optional[OrderedDict]:
        """Represents NLU training data to a dict/list structure ready to be
        serialized as YAML.

        Args:
            training_data: `TrainingData` to convert.

        Returns:
            `OrderedDict` containing all training data.
        """
        from rasa.shared.utils.validation import KEY_TRAINING_DATA_FORMAT_VERSION
        from ruamel.yaml.scalarstring import DoubleQuotedScalarString
        from rasa.shared.constants import (
            DOCS_URL_TRAINING_DATA,
            LATEST_TRAINING_DATA_FORMAT_VERSION,
        )

        result: OrderedDict[Text, Any] = OrderedDict()
        result[KEY_TRAINING_DATA_FORMAT_VERSION] = DoubleQuotedScalarString(
            LATEST_TRAINING_DATA_FORMAT_VERSION
        )

        intent_items = sorted(list(training_data.intents))
        if intent_items:
            result[KEY_INTENTS] = intent_items

        entity_items = sorted(list(training_data.entities))
        if entity_items:
            result[KEY_ENTITIES] = entity_items

        domain_items = []
        domain_items.extend(cls.process_domain_intents(training_data))
        # domain_items.extend(cls.process_synonyms(training_data))
        # domain_items.extend(cls.process_regexes(training_data))
        # domain_items.extend(cls.process_lookup_tables(training_data))

        if not any([domain_items, training_data.responses]):
            return None

        if training_data.responses:
            result[KEY_RESPONSES] = Domain.get_responses_with_multilines(
                training_data.responses
            )

        return result

    def dump_domain(
        self, target: Union[Text, Path, StringIO], training_data: "TrainingData"
    ) -> None:
        """Writes training data into a file in a YAML format.

        Args:
            target: Name of the target object to write the YAML to.
            training_data: TrainingData object.
        """
        result = self.domain_data_to_dict(training_data)

        if result:
            write_yaml(result, target, True)

class WatsonTrainingDataConverter(TrainingDataConverter):
    """Reads Watson training data and train a Rasa NLU model."""

    def _generate_path_for_converted_training_data(
        cls, source_file_path: Path, output_directory: Path
    ) -> Path:
        """Generates path for a training data file converted to YAML format.

        Args:
            source_file_path: Path to the original file.
            output_directory: Path to the target directory.

        Returns:
            Path to the target converted training data file.
        """
        return (
            output_directory / f"{source_file_path.stem}.yml", output_directory / f"{source_file_path.stem}_domain.yml"
        )

    def filter(self, source_path: Path) -> bool:
        """Checks if the given training data file Watson NLU Data.

        Args:
            source_path: Path to the training data file.

        Returns:
            `True` if the given file can be converted, `False` otherwise
        """
        if source_path.is_file:
            js = read_json_file(source_path)
            return self._check_watson_file(js)
        elif source_path.is_dir:
            for root, _, files in os.walk(source_path, followlinks=True):
                for f in sorted(files):
                    source_path = Path(root, f)
                    js = read_json_file(source_path)
                    return self._check_watson_file(js)

    def convert_and_write(self, source_path: Path, output_path: Path) -> None:
        """Converts Watson NLU data into Rasa NLU Data Format.

        Args:
            source_path: Path to the training data file.
        Returns:
            yaml file written to the output path
        """
        output_nlu_path, output_domain_path = self._generate_path_for_converted_training_data(
            source_path, output_path
        )
        js = read_json_file(source_path)
        training_data = self.get_training_data(js)
        all_entities = self._list_all_entities(js)
        # entities = set()
        for e in all_entities:
            entity = list(e.keys())[0]
            training_data.entities.add(entity)
        # training_data["entities"] = entities
        RasaYAMLWriterDomain().dump(output_nlu_path, training_data)
        RasaYAMLWriterDomain().dump_domain(output_domain_path, training_data)

    def _transform_entity_synonyms(
        self, synonyms: List[Dict[Text, Any]], known_synonyms: Optional[Dict[Text, Any]] = None
    ) -> Dict[Text, Any]:
        entity_synonyms = known_synonyms if known_synonyms else {}
        for s in synonyms:
            if "value" in s and "synonyms" in s and s["synonyms"] is not None:
                for synonym in s["synonyms"]:
                    entity_synonyms[synonym] = s["value"]
        return entity_synonyms

    def get_training_data(self, js: Dict[Text, Any], **kwargs: Any) -> TrainingData:
        """Loads training data stored in the IBM Watson data format."""
        training_examples = []
        entity_synonyms = self._entity_synonyms(js)
        entity_synonyms = self._transform_entity_synonyms(entity_synonyms)
        all_entities = self._list_all_entities(js)
        for intent in js.get("intents"):
            examples = intent.get("examples")
            if not examples:
                continue
            intent = intent.get("intent")
            for text in examples:
                utterance = text.get("text")
                example_with_entities = self._unpack_entity_examples(
                    text=utterance,
                    intent=intent,
                    training_examples=training_examples,
                    all_entities=all_entities,
                )
                if utterance in example_with_entities:
                    continue

                self._add_training_examples_without_entities(
                    training_examples, intent, utterance
                )

        return TrainingData(training_examples, entity_synonyms)

    @staticmethod
    def _add_training_examples_without_entities(
        training_examples: List, intent: str, text: str
    ) -> None:
        training_examples.append(Message(data={INTENT: intent, TEXT: text}))

    def _unpack_entity_examples(
        self, text: str, intent: str, training_examples: List, all_entities: List,
    ) -> List:
        examples_with_entities = []
        all_entity_names = set().union(*(d.keys() for d in all_entities))
        for entity_name in all_entity_names:
            if entity_name in text:
                examples_with_entities.append(text)
                values_of_entity = [
                    a_dict[entity_name]
                    for a_dict in all_entities
                    if entity_name in a_dict
                ]
                if values_of_entity is None:
                    continue
                for val in values_of_entity[0]:
                    entities = []
                    unpack_text = text.replace(entity_name, val.get("value")).replace(
                        "@", ""
                    )
                    start_index = unpack_text.index(val.get("value"))
                    end_index = start_index + len(val.get("value"))
                    entities.append(
                        {
                            "entity": entity_name,
                            "start": start_index,
                            "end": end_index,
                            "value": val.get("value"),
                        }
                    )
                    training_examples.append(
                        Message(
                            data={INTENT: intent, TEXT: unpack_text, ENTITIES: entities}
                        )
                    )

        return examples_with_entities

    @staticmethod
    def _list_all_entities(js: Dict[Text, Any]) -> List:
        all_entities = []
        entities = js.get("entities")
        if entities:
            for entity in entities:
                all_entities.append({entity.get("entity"): entity.get("values")})
        return all_entities

    def _entity_synonyms(self, js: Dict[Text, Any]) -> List:
        entity_synonyms = []
        entities = js.get("entities")
        if entities:
            for entity in entities:
                for val in entity.get("values"):
                    entity_synonyms.append(
                        {"value": val.get("value"), "synonyms": val.get("synonyms"),}
                    )
        return entity_synonyms

    @staticmethod
    def _check_watson_file(js: Dict[Text, Any]) -> bool:
        try:
            if js.get("metadata").get("api_version").get("major_version") == "v2":
                return True
            logger.debug("Currently Watson's API Version v2 file is only supported.")
            return False
        except Exception as e:
            logger.debug(e)
            return False

def main(argv):
    converter = WatsonTrainingDataConverter()
    input_file = sys.argv[1]
    logger.info(f"Converting {input_file}")
    converter.convert_and_write(Path(input_file), Path("."))
    # converter.convert_and_write(Path("/Users/greg/Dev/rasa/watson/customer-care.json"), Path("."))

if __name__ == "__main__":
   main(sys.argv[1:])