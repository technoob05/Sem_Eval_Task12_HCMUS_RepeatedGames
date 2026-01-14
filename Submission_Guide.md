# Participation

## How to participate

1. Sign up on Codabench and join the competition.
2. Update your team name in your profile (recommended for leaderboard display).
3. Download data from our dataset repo: [https://github.com/sooo66/semeval2026-task12-dataset/](https://github.com/sooo66/semeval2026-task12-dataset/)
4. Train on `train_data`, validate on `dev_data`, and prepare predictions for the active phase.
5. Submit your predictions to Codabench and wait for the score to appear.

## Submission format

Prepare a single file named `submission.jsonl` with one JSON object per line:

* `id`: question id
* `answer`: comma-separated options (e.g., `A` or `A,B`)

Example:

```jsonl
{"id": "q-2020", "answer": "A"}
{"id": "q-2021", "answer": "B,D"}
```

Rules:

* Use uppercase letters only (A, B, C, D).
* Order does not matter (`A,B` == `B,A`).
* Missing or invalid lines are scored as 0.

## Packaging and submission

1. Put `submission.jsonl` at the root of a folder named `submission`.
2. Zip the folder as `submission.zip`.
3. Upload `submission.zip` on Codabench.

## Phases

* Development Phase: submit predictions for the dev set.
* Final Phase: submit predictions for the hidden test set (limited submissions).

## Support

* Mailing list: [https://groups.google.com/g/semeval2026-task12](https://groups.google.com/g/semeval2026-task12)
