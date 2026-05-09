# ML Compass Scrum Sprint Plan

**Project:** ML Compass — Expert System for Machine Learning Model Selection
**Methodology:** Scrum
**Sprint Duration:** 2 weeks per sprint
**Total Sprints:** 5

---

# Sprint 1 — Research & Foundation

## Sprint Goal

Understand the domain, define the scope, and prepare the basic project structure.

## Tasks

### Research Team

* [ ] Research expert systems
* [ ] Research AutoML tools
* [ ] Compare Auto-sklearn, TPOT, H2O AutoML, Google AutoML, and Azure AutoML
* [ ] Identify gaps in existing AutoML tools
* [ ] Write literature review draft
* [ ] Define supported ML problem types

### Knowledge Engineer

* [ ] Draft ML problem decision tree
* [ ] Draft guided questions for users
* [ ] Draft initial expert rules
* [ ] Identify possible rule conflicts
* [ ] Prepare initial knowledge base structure

### Programmer

* [ ] Create Git repository
* [ ] Set up Python virtual environment
* [ ] Install required dependencies
* [ ] Create project folder structure
* [ ] Create basic Streamlit app
* [ ] Create initial JSON/YAML knowledge base file

### Project Manager

* [ ] Define sprint backlog
* [ ] Assign team responsibilities
* [ ] Prepare initial Gantt chart
* [ ] Schedule Scrum meetings
* [ ] Track blockers and risks

## Parallel Tasks

* [ ] Literature review can run while programmer sets up project environment
* [ ] Rule drafting can run while UI skeleton is created
* [ ] Gantt chart preparation can run while research is ongoing

## Sprint Deliverables

* [ ] Initial architecture draft
* [ ] Literature review draft
* [ ] Working Streamlit skeleton
* [ ] Initial rule list
* [ ] Initial knowledge base structure

---

# Sprint 2 — Core Expert System Engine

## Sprint Goal

Build the main decision-making system that recommends ML problem types and models.

## Tasks

### Knowledge Engineer

* [ ] Finalize classification rules
* [ ] Finalize regression rules
* [ ] Finalize clustering rules
* [ ] Finalize time-series rules
* [ ] Finalize basic NLP rules
* [ ] Validate rules with subject matter expert

### Programmer

* [ ] Build inference engine
* [ ] Build rule parser
* [ ] Connect inference engine with knowledge base
* [ ] Implement classification decision logic
* [ ] Implement regression decision logic
* [ ] Implement clustering decision logic
* [ ] Implement time-series decision logic
* [ ] Implement basic NLP decision logic

### UI Team

* [ ] Build guided question interface
* [ ] Build answer input flow
* [ ] Build page navigation flow
* [ ] Display detected problem type
* [ ] Display recommended models

### Subject Matter Expert

* [ ] Review expert rules
* [ ] Validate model recommendations
* [ ] Validate metric recommendations
* [ ] Provide correction notes

## Parallel Tasks

* [ ] UI development can happen while inference engine is being built
* [ ] Rule validation can happen while backend logic is implemented
* [ ] Knowledge base updates can happen while UI pages are prepared

## Sprint Deliverables

* [ ] Working recommendation system
* [ ] Guided questioning flow
* [ ] Initial ML model recommendations
* [ ] SME feedback notes

---

# Sprint 3 — Explanation Engine & Code Generator

## Sprint Goal

Make the system beginner-friendly by adding explanations, preprocessing advice, metrics, and starter code.

## Tasks

### Programmer

* [ ] Build explanation engine
* [ ] Build preprocessing recommendation engine
* [ ] Build metric recommendation logic
* [ ] Build code template generator
* [ ] Generate classification starter code
* [ ] Generate regression starter code
* [ ] Generate clustering starter code
* [ ] Generate NLP starter code

### Knowledge Engineer

* [ ] Write explanation templates
* [ ] Write preprocessing rules
* [ ] Write metric selection rules
* [ ] Define model explanation text
* [ ] Define beginner-friendly warning messages

### UI Team

* [ ] Create recommendation result page
* [ ] Create explanation section
* [ ] Create preprocessing suggestion section
* [ ] Create metric recommendation section
* [ ] Create starter code display section

### Testing Team

* [ ] Start unit testing for inference rules
* [ ] Test explanation output
* [ ] Test code generator output
* [ ] Test preprocessing recommendations

## Parallel Tasks

* [ ] Explanation engine can be built while UI result page is designed
* [ ] Code generator can be built while testing team writes test cases
* [ ] Knowledge engineer can write templates while programmer connects backend modules

## Sprint Deliverables

* [ ] Human-readable explanations
* [ ] Recommended preprocessing steps
* [ ] Recommended evaluation metrics
* [ ] Generated starter code
* [ ] First unit test results

---

# Sprint 4 — Validation, Conflict Handling & Testing

## Sprint Goal

Improve reliability, catch user mistakes, and validate the system with experts and end users.

## Tasks

### Programmer

* [ ] Add missing input validation
* [ ] Add invalid answer handling
* [ ] Add target column validation
* [ ] Detect low-cardinality numeric columns
* [ ] Detect boolean-like numeric columns such as 0/1
* [ ] Warn user when numeric column may actually be categorical
* [ ] Add rule conflict detection
* [ ] Add rule priority logic
* [ ] Add confidence score or certainty level

### Testing Team

* [ ] Write test cases for classification recommendations
* [ ] Write test cases for regression recommendations
* [ ] Write test cases for clustering recommendations
* [ ] Write test cases for NLP recommendations
* [ ] Write test cases for invalid input
* [ ] Write test cases for rule conflicts
* [ ] Perform integration testing
* [ ] Record bugs and issues

### End Users

* [ ] Test system with beginner ML students
* [ ] Collect usability feedback
* [ ] Collect confusion points
* [ ] Collect improvement suggestions

### Subject Matter Expert

* [ ] Perform expert validation testing
* [ ] Compare system recommendation with expert recommendation
* [ ] Identify incorrect rules
* [ ] Approve final rule corrections

## Parallel Tasks

* [ ] Testing can run while validation features are implemented
* [ ] End-user feedback can be collected while SME validation is ongoing
* [ ] Bug fixing can run alongside documentation updates

## Sprint Deliverables

* [ ] Stable validation module
* [ ] Rule conflict handling strategy
* [ ] Bug report
* [ ] Expert testing report
* [ ] End-user feedback report

---

# Sprint 5 — Finalization, Documentation & Demo

## Sprint Goal

Finalize the system, complete documentation, and prepare for submission or presentation.

## Tasks

### Documentation Team

* [ ] Finalize introduction section
* [ ] Finalize objectives section
* [ ] Finalize literature review section
* [ ] Finalize specific domain section
* [ ] Finalize SME section
* [ ] Finalize expert system architecture section
* [ ] Finalize implementation section
* [ ] Finalize testing section
* [ ] Finalize discussion/results section
* [ ] Finalize future work section
* [ ] Finalize conclusion section
* [ ] Format tables, figures, and diagrams
* [ ] Format references
* [ ] Prepare appendix

### Programmer

* [ ] Polish Streamlit UI
* [ ] Fix remaining bugs
* [ ] Improve error messages
* [ ] Improve recommendation display
* [ ] Improve starter code formatting
* [ ] Test final deployed version

### Project Manager

* [ ] Finalize Gantt chart
* [ ] Prepare presentation slides
* [ ] Prepare demo script
* [ ] Prepare project video if required
* [ ] Assign presentation roles
* [ ] Conduct final dry run

### Entire Team

* [ ] Perform final system testing
* [ ] Review final report
* [ ] Review final presentation
* [ ] Practice demo flow
* [ ] Submit final project

## Parallel Tasks

* [ ] Report writing can happen while programmer fixes bugs
* [ ] Presentation preparation can happen while documentation is finalized
* [ ] Demo practice can happen after core system is stable

## Sprint Deliverables

* [ ] Final expert system
* [ ] Final report
* [ ] Final Gantt chart
* [ ] Final presentation slides
* [ ] Final demo script
* [ ] Final submission package

---

# Product Backlog

## Must Have

* [ ] Guided question interface
* [ ] Rule-based inference engine
* [ ] Knowledge base
* [ ] Model recommendation
* [ ] Metric recommendation
* [ ] Preprocessing suggestion
* [ ] Explanation output
* [ ] Starter code generation

## Should Have

* [ ] Input validation
* [ ] Rule conflict handling
* [ ] Feature sanity checker
* [ ] SME validation
* [ ] End-user testing

## Could Have

* [ ] Confidence score
* [ ] Better UI design
* [ ] Export recommendation as PDF
* [ ] Save user session
* [ ] Add more ML problem types

## Won't Have for MVP

* [ ] Computer vision support
* [ ] Reinforcement learning support
* [ ] Full AutoML model training
* [ ] Automatic dataset detection
* [ ] LLM integration

---

# Scrum Ceremonies Checklist

## Sprint Planning

* [ ] Select sprint goal
* [ ] Select sprint backlog
* [ ] Assign task owners
* [ ] Estimate task difficulty
* [ ] Identify blockers

## Daily Standup

* [ ] Share what was completed yesterday
* [ ] Share what will be done today
* [ ] Share blockers

## Sprint Review

* [ ] Demo completed features
* [ ] Show sprint deliverables
* [ ] Collect feedback
* [ ] Update backlog

## Sprint Retrospective

* [ ] Discuss what went well
* [ ] Discuss what went wrong
* [ ] Discuss what to improve
* [ ] Decide next sprint improvement action

---

# Suggested Parallel Workstreams

| Workstream        | Main Owner               | Can Run In Parallel With     |
| ----------------- | ------------------------ | ---------------------------- |
| Literature Review | Research Team            | Project setup                |
| Rule Design       | Knowledge Engineer + SME | UI development               |
| Backend Logic     | Programmer               | Knowledge base writing       |
| UI Development    | UI Team                  | Inference engine development |
| Testing           | Testing Team             | Feature development          |
| Documentation     | Documentation Team       | Final bug fixing             |
| Presentation      | Project Manager          | Report formatting            |

---

# Definition of Done

A task is considered done when:

* [ ] The feature or document section is completed
* [ ] The work is reviewed by at least one team member
* [ ] Bugs or issues are recorded
* [ ] Required changes are fixed
* [ ] The task output is saved in the correct project folder
* [ ] The task is checked off in the Scrum board
