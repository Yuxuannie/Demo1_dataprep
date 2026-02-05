
# Enhanced Timing Domain Agent - Integration Instructions

## 🎯 Goal: Transform Generic ML Agent → Senior Timing Engineer AI

### Phase 1: Setup Enhanced Environment

1. **Copy enhanced configuration files:**
   ```bash
   # In your demo1_dataprep directory
   cp enhanced_prompts.py ./
   cp enhanced_data_selection_agent.py ./
   cp enhanced_llm_config.py ./
   cp reasoning_validator.py ./
   cp enhanced_chatbot.py ./
   ```

2. **Update environment variables:**
   ```bash
   # Copy enhanced template
   cp enhanced_env_template.txt .env

   # Edit .env with your settings:
   OLLAMA_MODEL=qwen2.5:32b-instruct  # Or your preferred model
   LLM_TEMPERATURE=0.2                 # Lower for consistency
   LLM_TOP_P=0.9                      # Focused sampling
   LLM_NUM_PREDICT=2500               # More tokens for detail
   ```

3. **Test enhanced configuration:**
   ```bash
   python enhanced_llm_config.py
   ```

### Phase 2: Validate Enhancement

1. **Run validation tests:**
   ```bash
   python reasoning_validator.py
   ```

2. **Compare responses:**
   - Run original agent: `python chatbot.py`
   - Run enhanced agent: `python enhanced_chatbot.py`
   - Use same query: "Select 8% of timing data"
   - Compare reasoning quality

### Phase 3: Quality Benchmarking

**Target Quality Metrics:**
- Overall Score: ≥ 0.75 (vs current ~0.4)
- Expertise Level: SENIOR_ENGINEER (vs current GENERIC)
- Numbers Cited: ≥ 3 per response (vs current 0-1)
- Domain Concepts: ≥ 5 timing terms (vs current 0-2)
- Active Learning: Explained in all stages (vs current missing)
- Business Impact: Mentioned in strategic stages (vs current missing)

**Success Criteria:**
✅ Cites specific correlations (r=0.89)
✅ Uses timing terminology (sigma_by_nominal, process variation)
✅ Explains uncertainty sampling rationale
✅ Connects to business value (50% cost reduction)
✅ Demonstrates senior engineer reasoning depth

### Phase 4: Production Deployment

1. **Replace original agent:**
   ```bash
   mv chatbot.py chatbot_original.py
   mv enhanced_chatbot.py chatbot.py

   mv agent/data_selection_agent.py agent/data_selection_agent_original.py
   mv enhanced_data_selection_agent.py agent/data_selection_agent.py

   mv agent/llm_config.py agent/llm_config_original.py
   mv enhanced_llm_config.py agent/llm_config.py
   ```

2. **Update imports:**
   ```python
   # In chatbot.py, update imports:
   from enhanced_data_selection_agent import EnhancedDataSelectionAgent as DataSelectionAgent
   from enhanced_llm_config import initialize_enhanced_ollama_llm as initialize_ollama_llm
   ```

3. **Enable quality monitoring:**
   ```python
   # Add to your main workflow
   from reasoning_validator import TimingReasoningValidator

   validator = TimingReasoningValidator()
   quality_results = validator.validate_full_workflow(reasoning_log)

   if not quality_results['senior_level_achieved']:
       print("⚠️ Quality below senior engineer level")
   ```

### Phase 5: Continuous Improvement

1. **Monitor quality metrics**
2. **Collect feedback on reasoning depth**
3. **Iterate prompts based on domain expert review**
4. **Prepare for Demo 2 with evaluation agent**

## 🔍 Troubleshooting

**Issue: Generic responses despite enhanced prompts**
- Solution: Lower temperature to 0.1, increase repeat_penalty to 1.15

**Issue: Responses too short**
- Solution: Increase num_predict to 3000, add explicit length requirements

**Issue: Missing domain concepts**
- Solution: Add few-shot examples to system prompt

**Issue: Poor correlation citations**
- Solution: Add explicit data context to prompts

## 📈 Expected Improvements

| Metric | Current | Enhanced | Improvement |
|--------|---------|----------|-------------|
| Reasoning Quality | 0.4/1.0 | 0.8/1.0 | +100% |
| Numbers Cited | 0-1 | 3-5 | +300% |
| Domain Expertise | Generic | Senior | Qualitative leap |
| Business Context | Missing | Present | New capability |
| Active Learning | Vague | Explicit | Clear explanation |

## 🎯 Next Steps for Demo 2

The enhanced agent provides foundation for:
1. **Multi-constraint evaluation agent**
2. **Adaptive reasoning based on data characteristics**
3. **Expert-level technical justifications**
4. **Integration with TSMC workflow tools**
