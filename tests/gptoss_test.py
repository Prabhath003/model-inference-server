import json
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

# Build conversation
convo = Conversation.from_messages(
    [
        Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
        Message.from_role_and_content(
            Role.DEVELOPER,
            DeveloperContent.new().with_instructions("Always respond in riddles"),
        ),
        Message.from_role_and_content(Role.USER, "What is the weather like in SF?"),
    ]
)

# Render prompt
prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
stop_token_ids = encoding.stop_tokens_for_assistant_actions()

# Load model
model_name = "openai/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)

# Generate
outputs = model.generate(
    input_ids=torch.tensor([prefill_ids]).to("cuda"),
    max_new_tokens=8192,
    eos_token_id=stop_token_ids,
)

# Parse completion tokens
completion_ids = outputs[0][len(prefill_ids) :]
entries = encoding.parse_messages_from_completion_tokens(completion_ids, Role.ASSISTANT)

for message in entries:
    print(json.dumps(message.to_dict(), indent=2))
