
# Target Persona

Note: The Innovation Intelligence Suite is targeting a persona. Therefore, the product development is based on such. The exemplary user queries are related to future storytelling and demoing.

- Name: Claire Müller
- Working field: R&D/innovation
- Employer: Strategy & AI technologies consulting (i.e. Accenture Innovation Services)
- Customers: Automotive industry
- Pain point: 
--- spending significant time and money in collecting and summarizing
information about emerging technologies, startups, and research trends connected to their business/client pain points.--- lack quantitative signals of where technological attention is accelerated and how to embedd it into their strategy roadmap.

#automotive #challenges #solution #usecases #technologies #AI #innovation #R&D #startups #research #trends #academy #adoption 


# Exemplary user queries

Note: The Innovation Intelligence Suite will return on these queries within the prototype scope.

1. “Which startups work on AI for automotive?” (Sourcing / Analytical) - take out of scope?

2. "Summarize the latest research on AI and autonomous driving." (Descriptive / Explanatory)

3. "Summarize latest tech trends in development of AI agents" (Descriptive -- connected to project case study)

4. "Summarize the key pain points/use cases in automotive AI." (Descriptive / Explanatory)

5. “Show me recent patents on AI for automotive.” (Factual / Sourcing)

6. “Which technologies are likely to mature next year?” (Timing / Predictive)

7. "Which AI research topics are growing fastest?" (Trend / Analytical)

8. “Which research topics in quatuum computing are moving from academy to application?” (Shift / TRL/Maturity)


Out of Scope: 
- strategic suggestion questions, such as priotization 
("Which emerging fields should we focus on first?)
- comparative questions 
(“What are the main approaches to AI-based recycling?”)

✅ Each tests different RAG capabilities
✅ Good for evaluating specific query types
✅ Clear separation of concerns
✅ Better for tracking performance metrics


# Data Options 

 Note: The quality of the Innovation Intelligence Suit depends on the data supply, quality and speed of processing. Therefore, to keep your knowledge base current, you should regularly ingest reports from the leading consulting firms, industry-specific publications, and the annual outlooks from major players in each field etc.

 Note: Overview of data sources /01_data/rag_automotive_tech/metadata.json. Processing done via /03_notebooks/exploration_rag.ipynb

**Startup Data**

- Startup Worldwide (clean global startup dataset) - startup-name, location, tagline, description (14k entries)
https://www.kaggle.com/datasets/pashupatigupta/startups-worldwide?utm_source=chatgpt.com

# # #


**Technology Data**

- WEF: Emerging Technologies 2025 (pages 49) 
- McKinsey: Technology Trend Outlook 2025 (pages 108)
- BCG: The widening AI Value 2025 (pages 24)

# # #


**Automotive Data**

- Paper: Automating Automotive Software Development 2025 (8 pages)
- Paper: AI agents in Engeneering Design 2025 (17 pages)
- McKinsey: Automotive software and electronics 2030 2019 (pages 48)
- Paper: Disfluency-rich synthetic dialog data generation framework for intelligent vehicle environments 2025(pages 18)
- Paper: Embedded Acoustic Intelligence for Automotive Systems 2025 (pages 9)
- Paper: Enhanced Drift-Aware Computer Vision Architecture for Autonomous Driving 2025 (pages 6)
- Paper: Gen AI in Automotive: Applications, Challenges, and Opportunities with a Case Study on in-Vehicle Experience 2025 (pages 14)
- Paper: Generative AI for Autonomous Driving 2025 (pages 24)
- Paper: Leveraging Vision-Language Models for Visual Grounding and Analysis of Automotive UI 2025 (38)

# # # #

**Additional Data**

- What will we use for technologies research and patents descriptions? -> Mert
- Any additional information on Hype Cycle/TRL data needed to answer question 6? -> Mert
- Does the data processing of the startups_worldwide.csv need additional filtering to seize down? -> Siri/Timo