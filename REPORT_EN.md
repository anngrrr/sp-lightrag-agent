# Comparative Analysis of LLM Responses in Regulatory QA: GraphRAG, Direct Model Call, and Google AI Overview

## Abstract
This paper presents a comparative analysis of three response generation modes for the domain of construction regulations: GraphRAG (generation grounded in graph context), LLM-only (direct model call without retrieval), and Google AI Overview (web search via Google's internal agent). The comparison is performed on a question about building floor-count criteria. Current results show that GraphRAG produces the most verifiable answer with linkage to a specific SP clause, LLM-only provides a more detailed but less stable text in terms of regulatory precision, and Google AI Overview provides a broad web summary with mixed source quality.

**Keywords:** GraphRAG, LLM, Retrieval-Augmented Generation, construction regulations, regulatory QA.

## 1. Introduction
Response generation systems for regulatory tasks must provide not only clear language but also stable factual correctness. In applied engineering QA scenarios, an error in term or criterion interpretation can lead to incorrect understanding of regulatory requirements. Therefore, it is important to compare model behavior in three modes:
- with a retrieval loop and structured domain context (GraphRAG);
- without external context (LLM-only);
- with web search and overview generation (Google AI Overview).

The goal of this study is to assess differences in accuracy, verifiability, and practical usefulness of responses to the same regulatory question.

## 2. Materials and Methods

### 2.1. Experimental environment
The study was conducted in the `sp-lightrag-agent` project (Python 3.12, local Ollama infrastructure). Model used: `ministral-3:8b`.

### 2.2. Compared modes
1. **GraphRAG** - invocation of the project's LightRAG retrieval pipeline followed by answer generation from retrieved graph context.
2. **LLM-only** - direct HTTP call to the model chat endpoint via `https://ollama.com`.
3. **Google AI Overview** - Google overview answer using an internal web search agent.

### 2.3. Control question
- For **GraphRAG**: `How to determine the number of floors in a building? What is considered a floor?`
- For **LLM-only** and **Google AI Overview**: `SNiP as a knowledge source to answer the question: How to determine the number of floors in a building? What is considered a floor?`

## 3. Current Results

### 3.1. GraphRAG result
The current GraphRAG answer is significantly shorter and relies on a specific regulatory source: **SP 332.1325800.2017, clause 3.1.38**. The answer explicitly provides formal criteria, including:
- floor openings greater than 60% of floor area;
- height below 1.8 m as a sign that the space is not classified as a floor.

Typical characteristics:
- stronger regulatory focus;
- explicit reference to source;
- less verbosity and fewer applied explanations.

### 3.2. LLM-only result
The current LLM-only answer is extensive and consultation-oriented. The model provides a broad list of criteria (heights, floor types, edge cases), but also uses heterogeneous regulatory references and generalized wording without strict linkage to one source.

Typical characteristics:
- high completeness and detail;
- mixing of regulatory and heuristic recommendations;
- no explicit traceability to a specific primary-source clause in the argument structure.

### 3.3. Google AI Overview result
The current Google AI Overview answer contains a broad thematic summary, separates the concepts of "number of floors" and "storey count," and lists floor types. At the same time, source quality is mixed: secondary and encyclopedic sources appear alongside profile regulatory links.

Typical characteristics:
- broad coverage and user-friendly structure;
- presence of web links and references to explanatory letters;
- heterogeneous reliability level of sources for strict regulatory QA.

### 3.4. Comparison
| Criterion | GraphRAG | LLM-only | Google AI Overview |
|---|---|---|---|
| Regulatory accuracy | High | Medium | Medium |
| Verifiability of claims | High | Low/medium | Medium |
| Completeness of explanation | Medium | High | High |
| Risk of generalization and inaccurate norms | Lower | Higher | Medium |
| Practical suitability for strict regulatory QA | High | Medium | Medium |

## 4. Full Response Quotations

### 4.1. Full GraphRAG response
```text
In accordance with **Code of Practice 332.1325800.2017** and regulatory requirements, a **floor** is defined as the part of a building between the top mark of a slab (or ground floor level) and the top mark of the slab (roof) located above it, forming a planning level of the building.

### **What is considered a floor?**
A floor is recognized as:
- **Regular floor** - the space between two horizontal structural elements (slabs), including residential, public, and technical premises.
- **Attic floor (mansard)** - a floor whose facade is fully or partially formed by a sloped surface (for example, a roof).

### **What is not considered a floor?**
According to **clause 3.1.38 of SP 332.1325800.2017**, the following are not considered floors:
1. **A planning level of a building**, if:
   - there are openings in the floor occupying more than 60% of its area;
   - the clear height between enclosing structures/ground is less than 1.8 m.
2. **Technical space** between load-bearing horizontal structures (for example, trusses with section height above 1.8 m), if it is not intended for residential, auxiliary, or public premises.

### **Example of determining storey count**
If a building has:
- **Main floor** (between first and second slabs) -> considered a floor.
- **Technical basement** with height below 1.8 m -> **not considered a floor**.
- **Attic (mansard)** with sloped walls -> considered a floor.

Thus, to determine the number of floors, it is necessary to consider **the presence of slabs, room heights, and their functional purpose**.

---
### **References**
- [1] SP 332.1325800.2017 (clause 3.1.38)
```

### 4.2. Full LLM-only response
```text
In **SNiP (Construction Norms and Rules)** and modern **SP (Codes of Practice)**, there are norms defining the concept of a **"floor"** and rules for determining it in buildings. The main regulatory documents governing this issue are:

### **1. Main regulatory documents**
#### **SP 54.13330.2016** *"Apartment Residential Buildings. Updated edition of SNiP 31-01-2003"*
- **Clause 3.1.1** defines a floor as:
  > *"A floor is a part of a building bounded by horizontal structures (slabs) and located between two floor levels (or between the floor level of the first floor and the floor level of the basement)."*
- **Clause 3.1.2** clarifies:
  > *"A basement (semi-basement floor) is considered a floor if its floor-to-ceiling height is at least 2.2 m (for residential buildings)."*

#### **SP 2.13130.2020** *"Fire protection systems. Evacuation routes and exits"*
- **Clause 4.2.1** defines a floor from a fire safety perspective:
  > *"A floor is a part of a building bounded by horizontal fire barriers (slabs), as well as semi-basement and basement floors if they are intended for permanent occupancy."*
- **Clause 4.2.2** clarifies that an **attic (mansard)** is considered a floor if the floor-to-ceiling height in the central part is at least **1.5 m** (for residential spaces).

#### **SP 118.13330.2012** *"Single-family Residential Buildings. Updated edition of SNiP 31-01-2003"*
- Similar to SP 54.13330.2016, defines a floor in individual residential houses.

#### **SP 52.13330.2016** *"Public Buildings. Updated edition of SNiP 31-06-2009"*
- **Clause 3.1.1** provides a general floor definition for administrative, commercial, and other buildings:
  > *"A floor is a part of a building bounded by slabs, including semi-basement and basement floors if they are intended for permanent occupancy."*

---

### **2. How to determine the number of floors in a building?**
According to the norms, a **floor** is counted by the following rules:

#### **A. Core criteria**
1. **Presence of a slab (horizontal structure)**
   - Each level separated from the adjacent one by a slab (reinforced concrete, wood, metal) is considered a floor.
   - **Exception**: Technical levels (for example, an attic without permanent occupancy) are not considered residential/public floors.

2. **Room height**
   - **Residential buildings (SP 54.13330.2016):**
     - **Basement/semi-basement** - floor-to-ceiling height **>= 2.2 m**.
     - **Attic (mansard)** - central height **>= 1.5 m** (for residential rooms).
   - **Public buildings (SP 52.13330.2016):**
     - Minimum room height **>= 2.1 m** (for permanently occupied rooms).

3. **Floor purpose**
   - If a level is **not intended for permanent occupancy** (for example, technical floor, basement without residential use), it is **not considered a floor** for fire safety and planning calculations.

#### **B. Special cases**
| Floor type | Condition to be recognized as a floor | Regulatory basis |
|-----------|----------------------------------------|------------------|
| **Basement/semi-basement** | Height >= 2.2 m (residential), >= 2.1 m (public) | SP 54.13330.2016, SP 52.13330.2016 |
| **Attic (mansard)** | Central height >= 1.5 m (residential) | SP 2.13130.2020 |
| **Technical floor** | Not considered a floor if no permanent occupancy | SP 54.13330.2016 |
| **Spaces with height < 1.5 m** | Not considered a floor (for example, crawl space) | SP 2.13130.2020 |

#### **C. Examples of floor count calculation**
1. **Residential building (SP 54.13330.2016)**
   - **Semi-basement (height 2.5 m)** -> **1 floor**
   - **First floor (height 3.0 m)** -> **2 floors**
   - **Second floor (height 2.8 m)** -> **3 floors**
   - **Attic (height 1.6 m in center)** -> **4 floors** (because >= 1.5 m)
   - **Loft (height 1.2 m, non-residential)** -> **not counted as a floor**

2. **Public building (SP 52.13330.2016)**
   - **Basement (height 2.0 m, with offices)** -> **1 floor**
   - **First floor (height 3.5 m)** -> **2 floors**
   - **Technical floor (height 2.5 m, but no occupants)** -> **not counted as a floor**

---

### **3. Conclusion**
- A **floor** is a **part of a building separated by a slab**, with **minimum height** (2.2 m for residential basements, 1.5 m for attics).
- **Technical floors without occupants** are not counted.
- **Number of floors** is determined by **horizontal structures** and **premises purpose**.

**Recommended documents for verification:**
✅ **SP 54.13330.2016** (apartment residential buildings)
✅ **SP 2.13130.2020** (fire safety)
✅ **SP 52.13330.2016** (public buildings)

If you have a specific case (for example, attic or basement), clarify its parameters for a precise answer.
```

### 4.3. Full Google AI Overview response
```text
The determination of storey count and number of floors in Russia is regulated by construction norms and rules (SNiP), in particular, the updated **SP 54.13330.2022 "Apartment Residential Buildings"** (Appendix A, formerly SNiP 31-01-2003) and **SP 118.13330.2012 "Public Buildings and Structures"**.
[NO FKR MKD Saint Petersburg +1]

According to regulatory documents, these are two different concepts:
1. **Number of floors** - counts all floors (underground, basement, semi-basement, above-ground, technical, attic).
2. **Storey count** - counts only above-ground floors.
[Cadastre.ru +1]

---

## What is considered a floor (terms and definitions)
According to SNiP (SP), a floor is part of the building space between two horizontal slabs where premises are located.
[Wikipedia]

#### Floors included in total number of floors:
* Above-ground floor: floor with room floor mark not below planned ground level.
* Semi-basement floor: floor where slab top is above average planned ground level by at least 2 m.
* Basement floor: floor with room floor mark below planned ground level by more than half of room height.
* Underground floor: floor where room floor mark is below planned ground level by full room height.
* Attic floor (mansard): floor in attic space whose facade is fully or partly formed by a sloped or broken roof surface.
* Technical floor: floor for engineering equipment and communications. It can be in lower (technical underground), middle, or upper (technical attic) part of the building.
[Cadastre.ru +5]

#### What is not considered a floor:
* Technical level lower than 1.8 m: if engineering network space has height below 1.8 m, it is not considered a floor.
* Mezzanine: an internal platform (in residential/public premises - no more than 40% of room area), not separated by slabs into full floors.
[Modern Scientific and Technological Academy ANO DPO "SNTA" +1]

---

## How to determine the number of floors in a building
According to letters from the Ministry of Economic Development and SP norms, the determination is made by the following algorithm:
1. Count all levels: sum all floors, including semi-basement, basement, technical, and attic floors.
2. Special case (technical attic): a technical floor above the top floor is not counted in storey count (above-ground), but is counted in total number of floors.
3. Calculation example: if the building has a basement, 2 residential floors, and an attic, then:
* Number of floors: 4 (1 basement + 2 residential + 1 attic).
* Storey count: 3 (2 residential + 1 attic).
[CNTD.ru +1]
Source: letters of the Ministry of Economic Development of the Russian Federation (in particular, 20.03.2013 No. OG-D23-1407), referring to SNiP 31-01-2003 and SP 54.13330.2011/2022.
[Consultant.ru]
```

## 5. Discussion
Comparison of current responses shows that GraphRAG provides the most source-disciplined result and is better suited for tasks requiring strict regulatory verifiability. LLM-only tends toward broad generalized explanation, which introduces mixed references and less controlled precision. Google AI Overview is useful as a broad contextual layer, but for strict regulatory QA it requires additional source filtering by legal and domain relevance.

A follow-up check showed that the clause reference was found correctly, but the SNiP number in the answer was incorrect. This number is indeed present on the same page, but as a local reference to a specific clause, not as the reference document number for the answer as a whole. This type of error can be reduced by prompt tuning: the model should prioritize the reference document number over any nearby number in the text.

For publication-grade regulatory QA tasks, this distinction is critical: verifiability and source discipline are usually more important than stylistic completeness. At the same time, GraphRAG responses may require an additional post-processing layer to improve readability and remove local stylistic inconsistencies.

## 6. Conclusion
Based on the analysis of current responses to an identical question:
- **GraphRAG** provides a shorter but better-grounded answer with explicit support from a specific regulatory clause.
- **LLM-only** provides a detailed text but is weaker in source discipline and stability of regulatory precision.
- **Google AI Overview** is useful as a contextual overview layer but does not replace domain validation due to heterogeneous web source quality.

Therefore, for construction-regulation tasks where verifiability is critical, **GraphRAG** is the preferred mode. LLM-only and Google AI Overview should be used as auxiliary layers for broader explanation and external cross-checking after the regulatory core is fixed.

## 7. Study limitations
- Analysis was performed on one control question.
- Evaluation is qualitative and does not include formal statistical testing.
- Independent expert validation of each claim against the full regulatory corpus was not performed.
