>
> # SemEval 2026 Task 12: Abductive Event Reasoning (AER)
>
> [Task Website](https://sites.google.com/view/semeval2026-task12/)[GitHub Repository](https://github.com/sooo66/semeval2026-task12-dataset)
>
> ## Announcement
>
> **[2025-12-31]** Codabench Competition Released.
>
> ## Introduction
>
> Understanding the causes behind real-world events is a fundamental aspect of human reasoning. While large language models (LLMs) have made substantial progress in tasks such as information extraction and summarization, they still face challenges in  **abductive reasoning** —inferring the most plausible cause of an observed outcome from incomplete and potentially noisy evidence.
>
> **Abductive Event Reasoning (AER)** is a SemEval 2026 shared task designed to evaluate this capability in realistic settings. Given a real-world event and a set of retrieved documents providing relevant background information, systems are required to identify the most plausible **direct cause** of the event.
>
> The task focuses on causal reasoning under uncertainty, where multiple explanations may appear plausible, but only some are supported by the available evidence. By emphasizing abductive inference grounded in textual context, AER aims to advance research on reasoning, robustness, and interpretability in language models.
>
> ## Task Description
>
> The AER task is formulated as a **multiple-choice question answering** problem.
> Each instance consists of:
>
> * **Event** : a short description of an observed real-world event
> * **Context** : a set of retrieved documents related to the event (including distractors)
> * **Options (A–D)** : four candidate explanations written as natural language sentences
>
> Among the four options:
>
> * One or more options may be correct
> * One option is always *“None of the others are correct causes.”*
>
> Systems must output the label(s) of the correct option(s) (e.g., `A,B`) based on reasoning over the provided context.
>
> The dataset spans multiple domains, including politics, finance, and public emergencies, and is constructed using a combination of LLM-assisted generation and human validation to ensure both plausibility and difficulty.
>
> ## Data access
>
> Please download the datasets from: [https://github.com/sooo66/semeval2026-task12-dataset/](https://github.com/sooo66/semeval2026-task12-dataset/)
>
> ## Evaluation Metric
>
> System performance is evaluated at the **instance level** using an exact and partial matching scheme over the predicted answer options.
>
> Let **G** denote the set of gold-standard correct options for an instance, and **P** denote the set of options predicted by the system.
>
> Each instance is scored as follows:
>
> * **1.0 (Full Match)** : if *P = G*
> * **0.5 (Partial Match)** : if *P* is a non-empty proper subset of *G* (i.e., the prediction covers at least one correct option and contains no incorrect options)
> * **0.0 (Incorrect)** : otherwise, including cases where the prediction contains any incorrect option or is empty
>
> The final system score is computed as the  **average score across all evaluation instances** .
>
> Predictions must contain only valid option labels (A–D). Invalid labels result in a score of 0 for the corresponding instance.
>
> ## Join us
>
> * Mailing list: [https://groups.google.com/g/semeval2026-task12](https://groups.google.com/g/semeval2026-task12)
>
> ## FAQs
>
> 1. Can I use closed-source models?
>    * Yes. Please describe them clearly in your system description paper.
> 2. Is there any restriction on the model knowledge cut-off date?
>    * No, but please specify it clearly in your system description paper.
> 3. Can I use additional datasets (e.g., publicly provided ones from other sources)?
>    * Yes. Please cite them in the system description paper.
> 4. How many submissions can we make as a team?
>    * You can make up to 5 submissions.
> 5. Can we work as a team?
>    * Yes, Codabench does not provide an option for that, but you can create a user account for the team.
> 6. How can we identify a team?
>    * Make one account for the team. That team can only make submissions from that account
> 7. Do I have to submit a system description paper?
>    * We strongly encourage it, but it is not mandatory.
> 8. Our system did not perform very well, should I still write a system description paper?
>    * Yes! We want insights from all participants—even if your system did not perform well. Negative results are still valuable.
> 9. Do I need to pay conference registration fees and/or attend SemEval for my paper to be published?
>    * No. It is not required to attend SemEval or pay registration fees for your paper to be published. However, if you want to attend, you must pay the attendance fee.
