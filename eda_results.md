# SemEval-2026 Task 12: Abductive Event Reasoning - EDA Report

---

## 1. Dataset Overview

| Split | Questions | Topics | Total Documents |
|-------|-----------|--------|-----------------|
| sample | 200 | 10 | 164 |
| train | 1819 | 36 | 775 |
| dev | 400 | 36 | 775 |
| test | 612 | 24 | 405 |


## 2. Sample Question (from train_data)

- **ID**: `q-201`
- **Topic ID**: `22`

**Target Event:**
> Iran launched ballistic missile attacks against Al Asad and Erbil air bases in Iraq used by US and coalition forces on Tuesday night.

**Options:**
- A: On December 29, US forces conducted airstrikes at five Kataib Hezbollah facilities in Iraq and Syria, killing at least 25 people.
- B: After 2006, Muhandis founded Kataib Hezbollah.
- C: On December 27, Kataib Hezbollah attacked the K1 military base near Kirkuk, killing an American contractor and wounding American and Iraqi personnel.
- D: A U.S. drone strike killed Iranian Maj. Gen. Qassem Soleimani and Abu Mahdi al-Muhandis after Muhandis picked up Soleimani from an airplane.

**Golden Answer:** `D`


## 3. Answer Distribution Analysis

**Total questions with answers:** 2419
- Single answer: 1369 (56.6%)
- Multiple answers: 1050 (43.4%)

**Individual Option Frequency:**
| Option | Count |
|--------|-------|
| A | 990 |
| B | 943 |
| C | 964 |
| D | 907 |

**Multi-answer Patterns (top 10):**
| Pattern | Count |
|---------|-------|
| A,C | 138 |
| B,C | 125 |
| A,D | 123 |
| A,B | 115 |
| C,D | 113 |
| B,D | 101 |
| A,B,D | 96 |
| A,B,C | 90 |
| A,C,D | 86 |
| B,C,D | 63 |


## 4. Text Length Analysis

**Target Event Length (characters):**
- Min: 32, Max: 209, Avg: 86.3

**Option Length (characters):**
- Min: 26, Max: 236, Avg: 80.4


## 5. Document Analysis

**Documents per Topic:** Min=14, Max=31, Avg=21.5

**Document Content Length (chars):** Min=61, Max=277968, Avg=7557

**Top 10 News Sources:**
| Source | Count |
|--------|-------|
| The New York Times | 55 |
| NPR | 34 |
| CNN | 34 |
| The Washington Post | 24 |
| Al Jazeera | 23 |
| CNBC | 19 |
| The Guardian | 14 |
| Los Angeles Times | 13 |
| Business Insider | 9 |
| Yahoo | 9 |


## 6. Topic Analysis

- Unique topics in train: 36
- Unique topics in dev: 36
- Unique topics in test: 24

- Topic overlap (train & dev): 36
- Topic overlap (train & test): 0


## 7. Sample Topics

1. **Topic ID 1:** UK holds Brexit referendum, decides to leave EU
2. **Topic ID 3:** Trump supporters storm US Capitol
3. **Topic ID 4:** Former Japanese PM Shinzo Abe assassinated
4. **Topic ID 2:** COVID-19 pandemic triggers global health crisis
5. **Topic ID 5:** Elon Musk acquires Twitter, implements major reforms


## 8. File Size Summary

| Split | Questions | Docs |
|-------|-----------|------|
| sample | 0.09 MB | 1.41 MB |
| train | 0.96 MB | 8.30 MB |
| dev | 0.21 MB | 8.30 MB |
| test | 0.33 MB | 3.65 MB |

**Total dataset size: 23.25 MB**