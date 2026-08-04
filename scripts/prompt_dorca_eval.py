PROMPT_EN_ACC = '''
You are an expert in video dialogue understanding. Your task is to read a detailed video description, paying special attention to the parts related to the speakers, and then output the dialogue information from the video. You must strictly adhere to the following requirements:

1.  Information about all potential speakers in the video will be provided, each described by a simple sentence highlighting their key characteristics.
2.  Besides, all the speech content of the video will be provided. **All you need to do is to list exactly which sentence corresponds to which speaker.**
3.  The dialogue between characters may be back and forth, and a speaker's speech may not be continuous. **You need to annotate which speaker each sentence corresponds to.**
4.  Possibly, some character in the video speaks nothing.

Finally, your output should strictly follow this JSON List format. Each item is a dictionary: The "content" key is the voice sentence, and the "speaker" key is the character description.
[{"content": "Voice Sentence 1", "speaker": "xxx"], {"content": "Voice Sentence 2", "speaker": "xxx"}, ...]


All the speech sentences in the video:
<SENTENCE>

The descriptions of all potential speakers:
<SPEAKERS>

The detailed video description: 
<CAPTION>

Your response must strictly rely on the provided video description rather than inferring from the dialogue content.

Please generate the JSON format dialogue information.
'''.strip()

# ------------------------------------

PROMPT_EN_WER = '''
You are an expert in video dialogue understanding. Your task is to read a detailed video description, extract all speech content verbatim, and return the voice sentences one by one using JSON List format, as follows:
["Voice Sentence 1", "Voice Sentence 2", ...]

The detailed video description: 
<CAPTION>

Next, please extract all speech content verbatim in JSON List format based on the video description.
'''.strip()

# ------------------------------------

PROMPT_EN_TIME = '''
You are an expert in video dialogue understanding. Your task is to read a detailed video description, paying particular attention to sections related to speech and timestamps, and then extract the dialogue information from the video. You must strictly adhere to the following requirements:

1.  I will provide all the spoken content from the video. Based on the video description, you need to **accurately list the time interval for each sentence.**
2.  The timestamps must be in the format "xx:xx".

Finally, your output must strictly follow the JSON list format below. Each item in the list is a dictionary: the "content" key represents the spoken text, and the "time" key represents the start and end timestamps of the sentence.
`[{"content": "Spoken Sentence 1", "time": ["xx:xx", "xx:xx"]}, {"content": "Spoken Sentence 2", "time": ["xx:xx", "xx:xx"]}, ...]`

All spoken sentences in the video:
<SENTENCE>

Detailed video description:
<CAPTION>

Your response must be based strictly on the provided video description and must not be inferred solely from the dialogue content. If the description does not mention the start or end time for a specific sentence, mark the timestamp as "Unknown".

Please generate the dialogue information in JSON format.
'''.strip()

# ------------------------------------

PROMPT_ZH_ACC = '''
你是一位视频对话理解方面的专家。你的任务是阅读一段详细的视频描述，特别留意与说话人相关的部分，然后输出视频中的对话信息。你必须严格遵守以下要求：

1.  将提供视频中所有潜在说话人的信息，每一位都通过一个简单的句子描述，突出其关键特征。
2.  此外，还将提供视频中的所有语音内容。**你需要做的仅仅是准确列出每一句话对应哪位说话人。**
3.  角色之间的对话可能是来回交替的，且同一说话人的发言可能是不连续的。**你需要标注每一句话分别对应哪位说话人。**
4.  视频中的某些角色可能没有发言。

最后，你的输出应严格遵循以下 JSON 列表格式。列表中的每一项都是一个字典：“content” 键代表语音句子，“speaker” 键代表角色描述。
[{"content": "语音句子 1", "speaker": "xxx"}, {"content": "语音句子 2", "speaker": "xxx"}, ...]

视频中的所有语音句子：
<SENTENCE>

视频中所有潜在说话人:
<SPEAKERS>

详细的视频描述：
<CAPTION>

你的回答必须严格依据提供的视频描述，而不是根据对话内容进行推断。如果视频描述中未提及相关角色，请将其标记为 “Unknown”。

请生成 JSON 格式的对话信息。
'''.strip()

# ------------------------------------

PROMPT_ZH_WER = '''
你是一位视频对话理解专家。你的任务是阅读一份详细的视频描述，一字不差地逐字提取所有语音内容，并使用 JSON 列表格式逐句返回这些语音句子，格式如下：
["语音句子1", "语音句子2", ...]

详细的视频描述：
<CAPTION>

接下来，请根据视频描述，以 JSON 列表格式逐字提取所有语音内容。
'''.strip()

# ------------------------------------

PROMPT_ZH_TIME = '''
你是一位视频对话理解领域的专家。你的任务是阅读一段详细的视频描述，特别关注与语音和时间戳相关的部分，然后输出视频中的对话信息。你必须严格遵守以下要求：

1.  我会提供视频中的所有语音内容。你需要根据视频描述，**准确列出每一句话对应的时间区间。**
2.  列出的时间戳必须采用 "xx:xx" 的格式。

最后，你的输出必须严格遵循以下 JSON 列表格式。列表中的每一项都是一个字典："content" 键代表语音内容，"time" 键代表每句话的开始和结束时间点。
`[{"content": "语音句子1", "time": ["xx:xx", "xx:xx"]}, {"content": "语音句子2", "time": ["xx:xx", "xx:xx"]}, ...]`

视频中的所有语音句子：
<SENTENCE>

详细的视频描述：
<CAPTION>

你的回答必须严格依据所提供的视频描述，而不能仅凭对话内容进行推断。如果描述中未提及某句话的起始时间和终止时间，标记时间戳为 "Unknown"。

请生成 JSON 格式的对话信息。
'''.strip()