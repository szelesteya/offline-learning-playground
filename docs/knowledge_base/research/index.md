# Research Knowledge Base

Collection of research papers, reading notes, presentations, and experiment scripts for offline reinforcement learning and world models.

!!! info "Prioritized Research Roadmap"
    View the full active reading backlog, prioritized reading order, and tracking matrix in the **[Research Reading Backlog](reading_backlog.md)**.
    
    - **Tier 1 (Core Foundations):** Park et al. (Value Bottleneck), Park et al. (Flow Q-Learning), Physical Intelligence ($\pi0.6$ & RL Token), Park et al. (OGBench).
    - **Tier 2 (Thesis & World Models):** AtomVLA, HIL-SERL, Dreamer 4, DPPO, Foresight Residual RL.
    - **Tier 3 (Citation Skims):** Clean Slate Offline RL, GrndCtrl.

## Completed Notes & Presentations

- [Prudencio et al. (2024): A Survey on Offline Reinforcement Learning](offline_rl_survey_prudencio_et_al_2024/slides.md)
- [Lu et al. (2022): Revisiting Design Choices in Offline Model-Based Reinforcement Learning](revisiting_design_choices_lu_2022/slides.md)
- [Fujimoto et al. (2019): Off-Policy Deep Reinforcement Learning without Exploration (BCQ)](bcq_fujimoto_et_al_2018/notes.md)
- [Park et al. (2024): Is Value Learning Really the Main Bottleneck in Offline RL?](value_learning_bottleneck_park_et_al_2024/notes.md)
- [Park et al. (2025): Flow Q-Learning](flow_q_learning_park_et_al_2025/notes.md)

---

## Directory Naming Standard

Each subdirectory represents a single paper using lowercase snake_case:

```text
<short_title>_<first_author>[_et_al]_<year>
```

- **`<short_title>`**: 2–4 concise keywords from the paper title (e.g., `revisiting_design_choices`, `conservative_q_learning`).
- **`<first_author>`**: Surname of the lead author (e.g., `lu`, `kumar`).
- **`[_et_al]`**: Added if there are 3+ authors (optional/recommended for clarity).
- **`<year>`**: 4-digit publication or preprint year.

**Examples**:
- `revisiting_design_choices_lu_2022` (or `revisiting_design_choices_lu_et_al_2022`)
- `conservative_q_learning_kumar_et_al_2020`
- `offline_rl_survey_prudencio_2023`

---

## Folder Contents Standard

Each paper directory may contain:

| File / Folder | Purpose |
| :--- | :--- |
| `slides.md` | Presentation deck formatted for Marp with Zotero citation anchors. |
| `notes.md` | Paper summary, key mathematical formulations, takeaways, and open questions. |
| `paper.pdf` | (Optional) Stored copy or link to the paper preprint. |
| `assets/` | Diagrams, architecture figures, and experimental plots. |
| `scripts/` | Minimal reproductions, environment checks, or toy experiments. |
| `related/` | Supplementary materials, reviews, or appendix notes. |
