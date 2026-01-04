# Healthcare AI Platform - Presentation

> A step-by-step journey through production-ready MLOps on Databricks
>
> **The Dual Story**: Healthcare AI Platform + AI-Assisted Development with Claude Code
>
> 90+ slides showing not just WHAT was built, but HOW it was built with human-AI collaboration

---

## Slide 1: The Challenge
**Why Healthcare AI is Hard**

- Not just model accuracy - need production infrastructure
- Data governance, ACID transactions, model serving, monitoring
- DIY approach: 18-24 months before first experiment

---

## Slide 2: The Vision
**Production MLOps in 3-4 Days**

- Pneumonia detection from chest X-rays
- Complete end-to-end: Train → Monitor → Improve
- Champion/Challenger A/B testing with feedback loops

---

## Slide 3: The Secret Sauce
**Three Enablers Working Together**

- Databricks: Pre-built MLOps infrastructure (16-31 months saved)
- Claude Code: Zero-click automation for SDLC
- Terraform: IaC + AI context dual purpose

---

## Slide 4: Databricks Value
**vs Building from Scratch (EC2/ECS)**

- Unity Catalog: 3-6 months saved
- Delta Lake (ACID on S3): 6-12 months saved
- Model Serving + A/B testing: 2-3 months saved

---

## Slide 5: Databricks Value (cont.)
**What You'd Need to Build**

- 10+ services: EMR, ECS, RDS, Kafka
- Custom governance, ACID implementation, inference logging
- Total: 16-31 months engineering effort eliminated

---

## Slide 6: Databricks Value - Data Ingestion
**Kaggle Integration in Minutes**

- 1000 X-ray images from Kaggle → Delta tables
- Built-in secrets management for API keys
- Minutes, not days of ETL pipeline building

---

## Slide 7: Claude Code Value
**Zero-Click Automation**

- Goal: No manual clicking in consoles
- AI automates full SDLC (Plan → Test → Document)
- This is DEV automation (not CI/CD or app automation)

---

## Slide 8: What is "No-Clicking"?
**Dev Automation vs Other Automation**

- Not CI/CD (that runs after code committed)
- Not application automation (end-user workflows)
- DEV automation: Conversation → Deployed infrastructure

---

## Slide 9: Terraform's Dual Purpose
**Innovation in AI-Assisted Development**

- Traditional: Infrastructure as Code
- Novel: Perfect structured context for AI
- Result: AI reads schemas, writes correct code

---

## Slide 10: DIY vs Databricks + Claude Code
**Timeline Comparison**

- DIY (EC2/ECS): 18-24 months to MVP
- Databricks + Claude Code: 3-4 days to MVP
- 100x faster time to experimentation

---

## Slide 11: DIY vs Databricks + Claude Code
**Cost Comparison**

- DIY: ~$800-1,250/month + 3-5 engineers
- Databricks: ~$160-350/month + zero ops burden
- Similar cost, massively less complexity

---

## Slide 12: Product Management Approach
**Issue-Driven Development**

- 15 GitHub issues tracked all work
- Human: Vision, architecture, strategic decisions
- AI: SDLC execution, documentation, deployment

---

## Slide 13: Strategic Outcomes
**What Was Achieved**

- Production MLOps: A/B testing, feedback loops
- Zero-click automation: 100% from conversation to cloud
- MLOps experimentation: Days, not months

---

## Slide 14: The Foundation - AWS Integration
**Secure, Scalable Cloud Infrastructure**

- Two S3 buckets: Unity Catalog + data lake
- IAM roles with cross-account access
- Everything defined as Infrastructure as Code (Terraform)

---

## Slide 15: Data Governance - Unity Catalog
**One Place to Rule Them All**

- Unified governance for data and AI assets
- Three-level namespace: catalog → schema → table/model
- Centralized access control and lineage tracking

---

## Slide 16: Data Architecture - Medallion Pattern
**Bronze → Silver → Gold**

- **Bronze**: Raw X-ray images (1000 files)
- **Silver**: Cleaned, validated data (future)
- **Gold**: Feedback tables and model performance metrics

---

## Slide 17: The Data - Real Medical Images
**Kaggle Chest X-Ray Pneumonia Dataset**

- 1000 X-ray images (500 normal, 500 pneumonia)
- Stored in Unity Catalog external volume
- Metadata tracked in Delta tables

---

## Slide 18: Data Ingestion - From Kaggle to Delta
**Automated Pipeline with Tracking**

- Download 1000 images from Kaggle API
- Store JPEGs in Unity Catalog volume
- Write metadata to Bronze Delta table

---

## Slide 19: The ML Question
**How Do We Build a Pneumonia Classifier?**

- Need fast POC to prove workflow
- 100 images sufficient for initial testing
- Simple CNN instead of complex models

---

## Slide 20: Training - TensorFlow Model (Champion)
**First Model: Keras CNN**

- Small CNN: 64x64 images, simple architecture
- Trains in 5-10 minutes on CPU
- Registered in MLflow with Unity Catalog

---

## Slide 21: Training - PyTorch Model (Challenger)
**Second Model: Educational Comparison**

- Same architecture, different framework (PyTorch)
- Shows framework flexibility and best practices
- Students learn both major ML frameworks

---

## Slide 22: MLflow Model Registry
**Versioned, Governed Model Storage**

- `pneumonia_poc_classifier` (TensorFlow/Keras)
- `pneumonia_poc_classifier_pytorch` (PyTorch)
- Unity Catalog integration for governance

---

## Slide 23: The Deployment Challenge
**From Notebook to Production API**

- How do doctors access predictions?
- Need real-time REST API endpoint
- Must handle multiple models simultaneously

---

## Slide 24: Model Serving - Single Endpoint
**REST API for Real-Time Predictions**

- Serverless endpoint (auto-scaling, cost-effective)
- REST API accessible from anywhere
- Handles cold starts (30-60s first request)

---

## Slide 25: The A/B Testing Problem
**How Do We Know Which Model is Better?**

- We have two models (Keras vs PyTorch)
- Can't just replace - need comparison
- Must track which model served each prediction

---

## Slide 26: Champion/Challenger Pattern
**The Industry Standard for Model Updates**

- **Champion**: Current production model (proven)
- **Challenger**: New model being tested (experimental)
- Gradual rollout: 50/50 → 70/30 → 90/10

---

## Slide 27: A/B Testing Endpoint
**Single Endpoint, Multiple Models**

- Traffic splitting: 50% Champion, 50% Challenger
- Databricks automatically logs which model served
- Inference tables capture every prediction

---

## Slide 28: Inference Table Logging
**Automatic Prediction Tracking**

- Every prediction automatically logged to Delta table
- Includes: request_id, served_model_name, response, timestamp
- No code changes needed (auto_capture)

---

## Slide 29: The Missing Piece - Ground Truth
**Predictions Alone Don't Tell Us Accuracy**

- Model says "PNEUMONIA" → Was it correct?
- Need radiologist to confirm diagnosis
- This happens hours or days later

---

## Slide 30: Feedback Infrastructure
**Closing the Loop with Ground Truth**

- `prediction_feedback` table stores radiologist diagnoses
- Links to predictions via request_id
- Tracks: true-positive, false-positive, true-negative, false-negative

---

## Slide 31: Feedback Collector API
**BentoML-Style Simple Interface**

- `submit_feedback(request_id, "true-positive", radiologist_id="DR001")`
- Maps feedback type to ground truth
- Writes to Delta table automatically

---

## Slide 32: The Complete Feedback Loop
**Connecting Predictions to Reality**

- **Step 1**: Make prediction, capture request_id
- **Step 2**: Doctor reviews, submits feedback
- **Step 3**: JOIN predictions + feedback tables

---

## Slide 33: Model Performance View
**Real-Time Accuracy Calculation**

- `model_performance_live` view JOINs inference + feedback
- Calculates per-model accuracy automatically
- Updates as new feedback arrives

---

## Slide 34: Monitoring Dashboard
**Champion vs Challenger Comparison**

- Traffic distribution (actual vs configured)
- Feedback coverage (% with ground truth)
- Accuracy comparison with confusion matrices

---

## Slide 35: Statistical Significance
**Is the Difference Real or Luck?**

- Chi-square test for significance
- P-value calculation (< 0.05 = significant)
- Sample size validation

---

## Slide 36: Automated Decision Engine
**Data-Driven Model Promotion**

- **PROMOTE**: Challenger significantly better (accuracy + p-value)
- **KEEP TESTING**: No significant difference yet
- **ROLLBACK**: Challenger significantly worse

---

## Slide 37: Demo Notebook - Two Approaches
**SDK vs REST API**

- **MLflow SDK**: Fast batch processing (1000s images)
- **REST API**: Real-time predictions (web/mobile apps)
- Same model, different access patterns

---

## Slide 38: Demo - Prediction Tracking
**Capturing the Request ID**

- Every REST API response includes `request_id`
- Save this ID for later feedback
- Links prediction to ground truth

---

## Slide 39: Demo - Submitting Feedback
**Simulating Radiologist Review**

- Import feedback_collector module
- Call `submit_feedback(request_id, feedback_type)`
- Data flows to Delta table

---

## Slide 40: Demo - Querying Performance
**Seeing Which Model is Winning**

- Query `model_performance_live` view
- See accuracy per model
- Make informed promotion decisions

---

## Slide 41: The Complete MLOps Cycle
**From Training to Continuous Improvement**

1. **TRAIN**: Keras + PyTorch models
2. **DEPLOY**: A/B testing endpoint (50/50)
3. **PREDICT**: REST API with request tracking

---

## Slide 42: The Complete MLOps Cycle (cont.)
**The Feedback Loop**

4. **COLLECT**: Radiologist submits ground truth
5. **ANALYZE**: JOIN predictions + feedback
6. **DECIDE**: Promote Challenger or keep Champion

---

## Slide 43: Why This Matters - Production Ready
**Not Just a POC, Real MLOps**

- Real-world accuracy tracking (not just validation set)
- Safe model updates with A/B testing
- Data-driven decisions with statistical rigor

---

## Slide 44: Why This Matters - Educational Value
**Teaching Modern MLOps**

- Complete end-to-end workflow demonstrated
- Industry-standard patterns (Champion/Challenger)
- Two frameworks (TensorFlow + PyTorch)

---

## Slide 45: Why This Matters - Scalability
**Built to Grow**

- Environment-based naming (dev → pilot → prod)
- Infrastructure as Code (reproducible)
- Terraform state management

---

## Slide 46: Technical Stack
**Modern, Industry-Standard Tools**

- **Cloud**: AWS (S3, IAM)
- **Platform**: Databricks on AWS
- **Governance**: Unity Catalog

---

## Slide 47: Technical Stack (cont.)
**MLOps & Storage**

- **Storage**: Delta Lake (ACID transactions)
- **ML**: MLflow + TensorFlow + PyTorch
- **Serving**: Databricks Model Serving

---

## Slide 48: Technical Stack (cont.)
**Deployment & Monitoring**

- **IaC**: Terraform (everything versioned)
- **Tracking**: GitHub Issues
- **Monitoring**: Custom feedback dashboard

---

## Slide 49: Key Achievements - Infrastructure
**Solid Foundation**

- ✅ AWS S3 integration with Unity Catalog
- ✅ Production Delta tables with Terraform
- ✅ 1000 X-ray images ingested

---

## Slide 50: Key Achievements - ML Pipeline
**End-to-End Workflow**

- ✅ Two trained models (Keras + PyTorch)
- ✅ MLflow registry with Unity Catalog
- ✅ Model serving endpoints deployed

---

## Slide 51: Key Achievements - A/B Testing
**Production MLOps**

- ✅ Champion/Challenger A/B testing infrastructure
- ✅ Feedback loop with ground truth
- ✅ Monitoring dashboard with statistical testing

---

## Slide 52: What Makes This Special - Completeness
**Every Piece of Production MLOps**

- Most demos stop at training
- This includes deployment, monitoring, improvement
- Real feedback loop with ground truth

---

## Slide 53: What Makes This Special - Best Practices
**Industry-Standard Patterns**

- Champion/Challenger (not just model replacement)
- Statistical significance testing (not gut feelings)
- Infrastructure as Code (reproducible)

---

## Slide 54: What Makes This Special - Governance
**Enterprise-Ready**

- Unity Catalog for centralized governance
- Delta Lake for ACID guarantees
- Terraform for audit trail

---

## Slide 55: The Meta-Story
**Zero-Click Automation Achieved**

- Project goal: Eliminate console clicking
- Claude Code with CLI access enabled this
- 100% automation from conversation to deployment

---

## Slide 56: Full SDLC Automation
**AI Across Entire Workflow**

- Plan → Build → Deploy → Test → Document
- Testing: AI runs OR guides with clear scenarios
- Everything versioned and reproducible

---

## Slide 57: Testing Strategy
**Two Approaches**

- AI runs automated tests based on requirements
- OR AI guides user with closed-ended test scenarios
- Pass/fail criteria match original user needs

---

## Slide 58: MLOps Experimentation Enabled
**Rapid Pattern Testing**

- A/B testing implemented in days
- Multiple frameworks (TensorFlow + PyTorch) easily compared
- Staff engineer experiments, AI automates execution

---

## Slide 59: Terraform's Dual Role
**Strategic Innovation**

- Infrastructure as Code (traditional)
- AI context builder (novel)
- Result: Correct code first try

---

## Slide 60: The Strategic Shift
**Human vs AI Roles**

- Human: Vision, architecture, strategic decisions
- AI: SDLC automation, deployment, documentation, testing
- Result: Engineer becomes architect, not typist

---

## Slide 61: Lessons Learned - Cold Starts
**Serverless Endpoints Need Warm-Up**

- First request: 30-60 seconds
- Solution: Dedicated warm-up request
- Trade-off: Cost vs latency

---

## Slide 62: Lessons Learned - Model Signatures
**Unity Catalog Requires Schemas**

- Must define input/output schema
- Use `infer_signature()` during training
- Enables validation and compatibility checks

---

## Slide 63: Lessons Learned - Payload Formats
**Different Frameworks, Different Expectations**

- TensorFlow/Keras: `{"inputs": [array]}`
- PyTorch: Varies by wrapper
- Always test with sample requests

---

## Slide 64: Future Enhancements - Better Models
**From POC to Production Quality**

- Transfer learning (EfficientNet, ResNet)
- Larger dataset (full 5000+ images)
- Hyperparameter tuning with AutoML

---

## Slide 65: Future Enhancements - Feature Engineering
**Improving Data Quality**

- Bronze → Silver transformation pipeline
- Data quality checks and validation
- Feature extraction and engineering

---

## Slide 66: Future Enhancements - Automation
**CI/CD and Auto-Retraining**

- GitHub Actions for deployment
- Automated retraining on feedback
- Real-time alerting for model drift

---

## Slide 67: The Journey Recap
**What We Built**

- From AWS buckets to production AI
- From raw images to actionable insights
- From single models to A/B testing

---

## Slide 68: The Journey Recap (cont.)
**How We Built It**

- Infrastructure as Code (Terraform)
- Issue-driven development (GitHub)
- Best practices throughout

---

## Slide 69: The Business Value
**Why This Matters to Healthcare**

- Safer AI deployment (A/B testing)
- Data-driven decisions (feedback loop)
- Continuous improvement (challenger promotion)

---

## Slide 70: The Educational Value
**Why This Matters to Students**

- Complete MLOps workflow demonstrated
- Both TensorFlow and PyTorch
- Real-world production patterns

---

## Slide 71: The Technical Value
**Why This Matters to Engineers**

- Reproducible infrastructure (Terraform)
- Governed assets (Unity Catalog)
- Scalable architecture (Databricks + AWS)

---

## Slide 72: Try It Yourself - Setup
**Getting Started**

1. Clone repository from GitHub
2. Configure AWS CLI and Databricks
3. Run `terraform apply` (5-7 minutes)

---

## Slide 73: Try It Yourself - Ingest Data
**Load the X-Ray Images**

1. Open `/Shared/ingest-kaggle-xray-data` notebook
2. Add Kaggle credentials to secrets
3. Run notebook (10-15 minutes)

---

## Slide 74: Try It Yourself - Train Models
**Create the Classifiers**

1. Run `/Shared/train-poc-model` (TensorFlow)
2. Run `/Shared/train-poc-model-pytorch` (PyTorch)
3. Check MLflow registry

---

## Slide 75: Try It Yourself - Deploy A/B Endpoint
**Set Up Champion vs Challenger**

1. Run `/Shared/deploy-ab-testing-endpoint` notebook
2. Wait 5-10 minutes for endpoint
3. Verify in Model Serving UI

---

## Slide 76: Try It Yourself - Make Predictions
**Test the System**

1. Run `/Shared/demo-model-usage` notebook
2. See predictions from both models
3. Capture request IDs

---

## Slide 77: Try It Yourself - Submit Feedback
**Close the Loop**

1. Use `submit_feedback()` function
2. Simulate radiologist ground truth
3. Query `model_performance_live` view

---

## Slide 78: Try It Yourself - Monitor Performance
**Compare Champion vs Challenger**

1. Run `/Shared/monitor-ab-test` notebook
2. View traffic distribution charts
3. See accuracy comparison and recommendations

---

## Slide 79: Resources - Documentation
**Where to Learn More**

- GitHub repository with all code
- 15 issues documenting every step
- README with detailed setup guide

---

## Slide 80: Resources - Notebooks
**Hands-On Learning**

- 9 Databricks notebooks included
- Commented code with explanations
- Educational notes throughout

---

## Slide 81: Resources - Community
**Built with Claude Code**

- Infrastructure managed with AI assistance
- Issue-driven development approach
- Every commit linked to issues

---

## Slide 82: Final Thoughts - Dual Value
**Two Achievements in One**

- Healthcare AI: Production MLOps platform
- Zero-click automation: SDLC workflow achieved
- Both demonstrable and reproducible

---

## Slide 83: The Technical Achievement
**Complete MLOps Platform**

- Train → Monitor → Improve cycle
- A/B testing for safe updates
- Feedback loop for continuous improvement

---

## Slide 84: The Process Achievement
**Zero-Click Automation**

- Goal: Eliminate console clicking
- Result: 100% automation achieved
- Staff engineer experiments, AI executes

---

## Slide 85: Call to Action - For Students
**Learn Modern MLOps**

- Run the notebooks yourself
- Experiment with different models
- Understand production patterns

---

## Slide 86: Call to Action - For Engineers
**Achieve Zero-Click Automation**

- Eliminate manual console operations
- Use AI for SDLC automation
- Focus on architecture, not clicking

---

## Slide 87: Call to Action - For Teams
**Deploy This to Production**

- Clone repository, customize for use case
- 100% Terraform managed (zero clicking)
- Benefit from proven architecture and workflow

---

## Slide 88: Thank You
**Questions?**

- Repository: github.com/rgasiorek/healthcare-ai-platform-databricks
- 100% automation achieved (zero clicking)
- Healthcare AI + SDLC automation demonstrated

---

## Slide 89: Key Takeaways
**Remember These Three Things**

1. Complete MLOps with feedback loops
2. Zero-click automation for development workflow
3. Human strategy + AI execution = 10x faster

---

## Appendix: Quick Reference
**Key Components**

| Component | Location | Purpose |
|-----------|----------|---------|
| Models | `healthcare_catalog_dev.models` | TensorFlow + PyTorch |
| Feedback | `gold.prediction_feedback` | Ground truth |
| Monitoring | `/Shared/monitor-ab-test` | Performance |

---

## Appendix: Commands
**Essential Operations**

```bash
# Deploy infrastructure
terraform apply

# Train models
# Run notebooks in /Shared/

# Monitor performance
# Run /Shared/monitor-ab-test
```

---

## Appendix: Architecture Diagram
**Complete System**

```
AWS S3 → Unity Catalog → Delta Tables
                ↓
          MLflow Registry
                ↓
         Model Serving (A/B)
                ↓
         Inference Tables
                ↓
         Feedback Loop
                ↓
         Monitoring Dashboard
                ↓
         Promotion Decision
```

---

**End of Presentation**

🤖 Created with [Claude Code](https://claude.com/claude-code)
