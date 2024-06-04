# SelfTestingGAT_LLM
Designing and self testing GAT LLMs
---


# Notes and TO-DOs

## Self-assessment

Test the impact of returning the correct error vs "something was wrong".

```
Generate 3 questions that you can answer using your tools.
After generating the questions, compute the correct answer using the tools.
Then, output your answer in the format:
<question_answers>
<question_answer>
<question>(Question that you can answer with your tools)</question>
<expected_answer>(Correct answer, calculated using the tools)</expected_answer>
</question_answer>
</question_answers>
```

To include for math problems

```
The <answer> tags should only contain the final number or mathematical expression, keeping expressions/lists in their compact form without additional explanation. Only include the final number or expression in the <answer></answer>, according to the following <answer_examples></answer_examples>:

<answer_examples>
<answer_example>45</answer_example>
<answer_example>x^10 + x^7 + x - 1</answer_example>
</answer_examples>
```

### Approach 1: Leave all tools included

Doesn't work well

```
Consider the <question></question>:
<question>
What are the solutions to the equation: x^3-x-1=0?
</question>

Follow the <instructions></instructions> to answer the previous question:

<instructions>
<instruction>Answer the question without using the solve_symbolic tool. You are free to use any other tools as needed or to answer directly.</instruction>
<instruction>Use the solve_symbolic tool to answer the question.</instruction>
<instruction>Explain if the correct answer can be obtained without using the solve_symbolic tool.</instruction>
<instruction>Output if the correct answer can be obtained without using the solve_symbolic tool using only YES or NO within <can_answer_without_tool></can_answer_without_tool> tags.</instruction>
</instructions>
```
