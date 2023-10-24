# Rasa Watson Converter

Script to convert a Watson chatbot export to Rasa 3.x yml format.

This is a fork of a [script](https://github.com/souvikg10/rasa_nlu/blob/watson-emulate/rasa/nlu/training_data/converters/watson_nlu_json_to_yaml_converter.py) by [souvikg10](https://github.com/souvikg10).

```sh
python watson_converter.py customer-care.json
```

In the example above, the script will create two files:

- `customer-care_domain.yml` - A Rasa domain file with a list of intents
- `customer-care.yml` - A Rasa NLU with the intents and example utterances as well as any synonyms`

## Sample Watson JSON

The sample json files come from github repos (search using [github](https://github.com/search?q=watson+%22workspace_id%22+%22learning_opt_out%22+language%3Ajson&type=code))

- [customer-care](https://github.com/watson-developer-cloud/assistant-skill-analysis/blob/master/tests/resources/test_workspaces/skill-Customer-Care-Sample.json)
