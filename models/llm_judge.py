from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
import torch
import re

# login first
from huggingface_hub import login
login(token="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")  # my hugging face read-only access token

# obtain model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

def call_llm_api_single(prompts, max_new_tokens=30):
    responses = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        responses.append(response_text)
    return responses


def construct_prompt(scene_label, candidate_labels):
    # context_prefix = (
    #     "You are a detailed and realistic scene analyst. Your task is to estimate the likelihood "
    #     "of objects appearing in different scenes based strictly on real-world common sense and typical scene compositions. "
    #     "Avoid being fair, equal, or arbitrary. Your estimates should reflect genuine and accurate real-world probabilities."
    # )

    # prefers scene-specific objects over more general ones
    # one-shot without reasoning
    # context_prefix = (
    #     "You are a detailed, knowledgeable, and realistic scene analyst. "
    #     "Your task is to evaluate the likelihood of objects appearing in a specific given scene. "
    #     "Objects uniquely and strongly associated with the particular scene context must be rated significantly higher. "
    #     "Common or generic objects that frequently appear across many unrelated contexts must receive lower ratings. "
    #     "For instance, in a 'library' scene, the object 'bookshelf' (highly context-specific) should clearly receive a higher rating compared to a generic object like 'table'. "  # one-shot.
    #     "Your ratings must reflect accurate, realistic real-world specificity and usage. Avoid equal or overly fair ratings."
    # )

    # one-shot with reasoning
    context_prefix = (
        "You are a detailed, knowledgeable, and realistic scene analyst. "
        "Your task is to determine how specifically and typically each object belongs to the given scene. "
        "Strongly prefer items uniquely or especially associated with that scene's function or activities, and give them significantly higher probabilities. "
        "Generic or broadly common items appearing in many places should have lower probabilities."
        "Your ratings must reflect accurate, realistic real-world specificity and usage. Avoid equal or overly fair ratings."
        "Give percentage probabilities in the range between 1 and 99 for each object, separated by ',' in terms of formatting.\n\n"

        "Example:\n"  # must align with prompt!
        "Your task: given the scene 'library', how relatively likely are \"bookshelf\", \"table\", \"sandcastle\", being in the scene, respectively?\n"
        "Scene: 'library'\n"
        "Objects: 'bookshelf', 'table', 'sandcastle'\n"
        "Reasoning: Both bookshelf and table are context-specific to the library, but table is more commonly found around places, while bookshelf is more unique to the setting so bookshelf is more preferred. "
        "Meanwhile, it is very unlikely for a sandcastle to appear in a library.\n"
        "Probabilities: bookshelf (90%), table (70%), sandcastle (1%)"  # purposefully degrading general objects
    )

    prompt = (
        f"{context_prefix}\n\n"
        f"Your task: given the scene '{scene_label}', how relatively likely are "
        + ", ".join(f"'{label}'" for label in candidate_labels)
        + " being in the scene, respectively?" # Give percentage probabilities in the range between 1 and 99 for each object, separated by ','."
    )
    return prompt

def call_llm_api_multiple(prompt, max_new_tokens=300):  # increased to 300! (should be enough hopefully)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=tokenizer.eos_token_id)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response_text

def extract_numeric_scores_single(responses, default=0.5):
    numeric_scores = []
    for response in responses:
        match = re.findall(r"\d+\.\d+|\d+", response)
        if match:
            # we take the first valid numerical match
            numeric_score = float(match[0])
            numeric_scores.append(numeric_score)
        else:
            # assign the default probability if parsing encounters error
            numeric_scores.append(default)
    return numeric_scores

def extract_numeric_scores_multiple(response, num_candidates, default=50.0):
    # Outdated number-grabbing
    # match = re.findall(r"\d+\.\d+|\d+", response)
    # if len(match) == num_candidates:
    #     return [float(score) for score in match]
    # else:
    #     return [default] * num_candidates
    
    # v4's:
    probability_matches = re.findall(r'Probabilities:(.*)', response)
    if probability_matches:
        # take the last occurrence (actual task's results)
        last_probabilities = probability_matches[-1]
        # extract numbers from within parentheses, ex "chair (80%)"
        match = re.findall(r'\((\d+\.?\d*)%\)', last_probabilities)
        # fallback if parentheses format isn't matched (i.e. not enough candidate scores)
        if len(match) != num_candidates:
            # attempt to extract plain numbers
            match = re.findall(r'\d+\.\d+|\d+', last_probabilities)
        else:
            # we are all good
            return [float(score) for score in match]

    # default fallback
    return [default] * num_candidates # a bunch of 50.

if __name__ == "__main__":
    scene_label = "hospital"
    candidate_labels = ["chair", "dog", "wheelchair"]
    '''
    Notes:
    v1: (from v0)
    Cannot do 'a numerical prob.... between 0.0 and 1.0' because the LLM returns 0 often. So it is better to give it more degrees of freedom with "percentage" and apply a soft boundary "1 and 99"
    so the LLM cannot be overly uncertain or confident.
    However, the model is still outputting minimal, lowerbound probability, like 1. It's hard to rank the likelihood of an item existing in a scene in a general setting! So, similar to constrative
    learning, instead of asking f"Given the scene '{scene_label}', how likely is '{label}' being in the scene? Give a percentage probability in the range between 1 and 99." once for each object label in scene,
    we will ask: f"Given the scene '{scene_label}', how relatively likely are '{label_1}', ..., '{label_n}' being in the scene, respectively? Give a percentage probability in the range between 1 and 99 for each object, separated by comma."
    Also with larger LMs we are testing, it is better to do one-pass call instead of multiple, in terms of run time and compute usage.
    v2:
    However, even with Flan-t5-xl, the model is still performing suboptimally (though better in output format). Maybe the model is trying to be fair! So we add a context prefix to our prompt trying to place the LM
    in a setting that can remove the restrictions on fairness.
    Sadly, we think Flan is just not it. So we went to HuggingFace and setup Mistral-7B-Instruct-v0.2, and it performs much better!
    v3:
    However, the LM still prefers chair over wheelchairs for common objects in the hosptial. While this kind of makes sense since a hospital probably contains a lot more chairs for people to sit on instead of wheelchairs for the patients,
    we would not prefer this since chair is a more general object that can exist across a lot more contexts than wheelchair, and chair is less scarce than wheelchair! So, we want to highlight context-specific scarce object. Therefore,
    we modified the in-context learning prompt once more to instruct the model what to prioritize.
    Sadly, with zero-shot in-context learning, the model's output is still suboptimal. For example: Chair: 85%, Wheelchair: 50%, Dog: 10%. So, we attempt to do one-shot in-context learning.
    v4:
    We added reasoning and aligned the task example and run-time task's formats so the model can follow the chain of thought better. We also added a chain of thought reasoning to an analogous scenario of how to prefer scene-specific
    object over more general ones. Thus, we were abhle to achieve the desired outcome on the chair vs wheelchair example. However, since the input is too long, the output now repeats the input prompt.
    To mitigate this issue, we updated the numeric scores extractor to go backwards from the response so the run-time task's output can be correctly extracted. An alternative would be shortening the prompt.


    Flan-t5 (google/flan-t5-xl): google's seq2seq
    Mistral (mistralai/Mistral-7B-Instruct-v0.2): Causal LM

    Logs:
    Flan-t5-base:
        single v1 - bad; ranking each object as minimum (0)
        multiple v1 - bad; ranking ALL objects with same probability '1' so default choice is outputted.
    Flan-t5-large:
        multiple v1 - bad; ranking ALL objects with same probability '1' so default choice is outputted.
    Flan-t5-xl: (takes longer)
        multiple v1 - bad; ranking ALL objects as same probability (11%).
        multiple v2 - bad; ranking ALL objects as same probability (11%).
    mistralai/Mistral-7B-Instruct-v0.2:
        multiple v2 - better. For example: Based on real-world common sense and typical hospital scenes, the probabilities would be as follows: chair: 95%, wheelchair: 85%, dog: 5%. 
        multiple v3 - okay.... but still can't learn to prefer wheelchair. Also introduced prompt-repeating in the output.
        multiple v4 - works! Has prompt-repeating but it's fine if we increase output max token count. And learns to prefer wheelchair correctly with one-shot in-context learning.
            i.e. 
            Scene: 'hospital'
            Objects: 'chair', 'wheelchair', 'dog'
            Reasoning: Both chair and wheelchair are context-specific to the hospital, but wheelchair is more directly associated with the hospital's function, making it more likely to be present. 
            Dogs, while not unheard of in hospitals, are less common and more variable, so they have a lower probability.
            Probabilities: chair (80%), wheelchair (90%), dog (20%)
            Numeric Scores: [80.0, 90.0, 20.0]

            Scene: 'hospital'
            Objects: 'chair', 'dog', 'wheelchair'
            Reasoning: Chairs are a common piece of furniture found in hospitals, but wheelchairs are more context-specific to the setting. Dogs are not typically found in hospitals, but they can be present in certain areas like therapy dogs.
            Probabilities: chair (80%), dog (10%), wheelchair (90%)
            Numeric Scores: [80.0, 10.0, 90.0]

            *this shows that the rating remains correctly even with a different ordering of input objects! this shows that there is no position-bias in the one-shot in-context learning, and that this prompt + LM
            is position invariant.
    '''
    # prompts = [f"Given the scene '{scene_label}', how likely is '{label}' being in here? Give a percentage probability in the range between 1 and 99." for label in candidate_labels]
    # response = call_llm_api_single(scene_label, candidate_labels)
    # numeric_scores = extract_numeric_scores_single(response, len(candidate_labels))

    prompt = construct_prompt(scene_label, candidate_labels)
    response = call_llm_api_multiple(prompt)
    numeric_scores = extract_numeric_scores_multiple(response, len(candidate_labels))

    print("Model:", model_name)
    print("Prompt:", prompt)
    print("Prompt Response:", response)
    print("Numeric Scores:", numeric_scores)