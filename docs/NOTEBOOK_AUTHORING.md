# Notebook authoring guide

## File naming

`notebooks/ch{NN}_{slug}.ipynb` — e.g. `ch03_agent_loop.ipynb`.

## Mandatory sections (H2 headings)

1. Chapter objectives  
2. Prerequisites  
3. Concept: \<topic\>  
4. How it works (+ mermaid)  
5. Reference implementation map (Harness Agent vs external harnesses)  
6. Design choices in harness_agent  
7. Implementation walkthrough  
8. Trace one request  
9. Hands-on exercise  
10. Common pitfalls  
11. Checkpoint questions  
12. Summary & next chapter  

## Branding

- Our code: `harness_agent`, `harness-agent`, `HARNESS_AGENT_HOME`.  
- Concept sections may name Hermes, OpenClaw, IDE agents for comparison.  

## Regenerating notebooks

```bash
python scripts/generate_notebooks.py
```
