# Healthcare AI Platform - Presentation

> A step-by-step journey through production-ready MLOps on Databricks
>
> **The Dual Story**: Healthcare AI Platform + AI-Assisted Development with Claude Code
>
> 90+ slides showing not just WHAT was built, but HOW it was built with human-AI collaboration

---

## Slide 1: The Challenge
**Building Healthcare AI That Actually Works in Production**

- Medical images need accurate, reliable AI diagnosis
- Traditional ML projects fail without proper infrastructure
- Production requires monitoring, updates, and continuous improvement

---

## Slide 2: The Vision
**From Data Lake to Self-Improving AI**

- Pneumonia detection from chest X-rays
- Complete end-to-end MLOps pipeline
- Champion/Challenger A/B testing with feedback loops

---

## Slide 3: The Secret Ingredient
**Built with AI-Assisted Development**

- Claude Code running locally with system access
- Direct GitHub, AWS, Databricks CLI access
- Beyond code: issues, deploys, docs, thinking partner

---

## Slide 4: What Makes This Unique
**The Most Important Story**

- Not just "here's a healthcare AI platform"
- But "here's HOW AI helped build it"
- Staff engineer + AI collaboration pattern

---

## Slide 5: Claude Code's Role
**AI as Full Development Partner**

- Created all 15 GitHub issues
- Wrote Terraform, notebooks, deployed infrastructure
- Documented, tested, debugged, closed tickets

---

## Slide 6: The Terraform Dual Purpose
**IaC AND AI Context**

- Traditional: Infrastructure as Code
- Novel: Perfect context for AI decisions
- AI reads schema, writes correct queries first try

---

## Slide 7: Value for Staff Engineers
**10x Faster, Not Replacing Engineers**

- Strategic control stays with human
- AI handles tactical execution
- Complete documentation burden lifted

---

## Slide 8: The Foundation - AWS Integration
**Secure, Scalable Cloud Infrastructure**

- Two S3 buckets: Unity Catalog + data lake
- IAM roles with cross-account access
- Everything defined as Infrastructure as Code (Terraform)

---

## Slide 9: Data Governance - Unity Catalog
**One Place to Rule Them All**

- Unified governance for data and AI assets
- Three-level namespace: catalog → schema → table/model
- Centralized access control and lineage tracking

---

## Slide 5: Data Architecture - Medallion Pattern
**Bronze → Silver → Gold**

- **Bronze**: Raw X-ray images (1000 files)
- **Silver**: Cleaned, validated data (future)
- **Gold**: Feedback tables and model performance metrics

---

## Slide 6: The Data - Real Medical Images
**Kaggle Chest X-Ray Pneumonia Dataset**

- 1000 X-ray images (500 normal, 500 pneumonia)
- Stored in Unity Catalog external volume
- Metadata tracked in Delta tables

---

## Slide 7: Data Ingestion - From Kaggle to Delta
**Automated Pipeline with Tracking**

- Download 1000 images from Kaggle API
- Store JPEGs in Unity Catalog volume
- Write metadata to Bronze Delta table

---

## Slide 8: The ML Question
**How Do We Build a Pneumonia Classifier?**

- Need fast POC to prove workflow
- 100 images sufficient for initial testing
- Simple CNN instead of complex models

---

## Slide 9: Training - TensorFlow Model (Champion)
**First Model: Keras CNN**

- Small CNN: 64x64 images, simple architecture
- Trains in 5-10 minutes on CPU
- Registered in MLflow with Unity Catalog

---

## Slide 10: Training - PyTorch Model (Challenger)
**Second Model: Educational Comparison**

- Same architecture, different framework (PyTorch)
- Shows framework flexibility and best practices
- Students learn both major ML frameworks

---

## Slide 11: MLflow Model Registry
**Versioned, Governed Model Storage**

- `pneumonia_poc_classifier` (TensorFlow/Keras)
- `pneumonia_poc_classifier_pytorch` (PyTorch)
- Unity Catalog integration for governance

---

## Slide 12: The Deployment Challenge
**From Notebook to Production API**

- How do doctors access predictions?
- Need real-time REST API endpoint
- Must handle multiple models simultaneously

---

## Slide 13: Model Serving - Single Endpoint
**REST API for Real-Time Predictions**

- Serverless endpoint (auto-scaling, cost-effective)
- REST API accessible from anywhere
- Handles cold starts (30-60s first request)

---

## Slide 14: The A/B Testing Problem
**How Do We Know Which Model is Better?**

- We have two models (Keras vs PyTorch)
- Can't just replace - need comparison
- Must track which model served each prediction

---

## Slide 15: Champion/Challenger Pattern
**The Industry Standard for Model Updates**

- **Champion**: Current production model (proven)
- **Challenger**: New model being tested (experimental)
- Gradual rollout: 50/50 → 70/30 → 90/10

---

## Slide 16: A/B Testing Endpoint
**Single Endpoint, Multiple Models**

- Traffic splitting: 50% Champion, 50% Challenger
- Databricks automatically logs which model served
- Inference tables capture every prediction

---

## Slide 17: Inference Table Logging
**Automatic Prediction Tracking**

- Every prediction automatically logged to Delta table
- Includes: request_id, served_model_name, response, timestamp
- No code changes needed (auto_capture)

---

## Slide 18: The Missing Piece - Ground Truth
**Predictions Alone Don't Tell Us Accuracy**

- Model says "PNEUMONIA" → Was it correct?
- Need radiologist to confirm diagnosis
- This happens hours or days later

---

## Slide 19: Feedback Infrastructure
**Closing the Loop with Ground Truth**

- `prediction_feedback` table stores radiologist diagnoses
- Links to predictions via request_id
- Tracks: true-positive, false-positive, true-negative, false-negative

---

## Slide 20: Feedback Collector API
**BentoML-Style Simple Interface**

- `submit_feedback(request_id, "true-positive", radiologist_id="DR001")`
- Maps feedback type to ground truth
- Writes to Delta table automatically

---

## Slide 21: The Complete Feedback Loop
**Connecting Predictions to Reality**

- **Step 1**: Make prediction, capture request_id
- **Step 2**: Doctor reviews, submits feedback
- **Step 3**: JOIN predictions + feedback tables

---

## Slide 22: Model Performance View
**Real-Time Accuracy Calculation**

- `model_performance_live` view JOINs inference + feedback
- Calculates per-model accuracy automatically
- Updates as new feedback arrives

---

## Slide 23: Monitoring Dashboard
**Champion vs Challenger Comparison**

- Traffic distribution (actual vs configured)
- Feedback coverage (% with ground truth)
- Accuracy comparison with confusion matrices

---

## Slide 24: Statistical Significance
**Is the Difference Real or Luck?**

- Chi-square test for significance
- P-value calculation (< 0.05 = significant)
- Sample size validation

---

## Slide 25: Automated Decision Engine
**Data-Driven Model Promotion**

- **PROMOTE**: Challenger significantly better (accuracy + p-value)
- **KEEP TESTING**: No significant difference yet
- **ROLLBACK**: Challenger significantly worse

---

## Slide 26: Demo Notebook - Two Approaches
**SDK vs REST API**

- **MLflow SDK**: Fast batch processing (1000s images)
- **REST API**: Real-time predictions (web/mobile apps)
- Same model, different access patterns

---

## Slide 27: Demo - Prediction Tracking
**Capturing the Request ID**

- Every REST API response includes `request_id`
- Save this ID for later feedback
- Links prediction to ground truth

---

## Slide 28: Demo - Submitting Feedback
**Simulating Radiologist Review**

- Import feedback_collector module
- Call `submit_feedback(request_id, feedback_type)`
- Data flows to Delta table

---

## Slide 29: Demo - Querying Performance
**Seeing Which Model is Winning**

- Query `model_performance_live` view
- See accuracy per model
- Make informed promotion decisions

---

## Slide 30: The Complete MLOps Cycle
**From Training to Continuous Improvement**

1. **TRAIN**: Keras + PyTorch models
2. **DEPLOY**: A/B testing endpoint (50/50)
3. **PREDICT**: REST API with request tracking

---

## Slide 31: The Complete MLOps Cycle (cont.)
**The Feedback Loop**

4. **COLLECT**: Radiologist submits ground truth
5. **ANALYZE**: JOIN predictions + feedback
6. **DECIDE**: Promote Challenger or keep Champion

---

## Slide 32: Why This Matters - Production Ready
**Not Just a POC, Real MLOps**

- Real-world accuracy tracking (not just validation set)
- Safe model updates with A/B testing
- Data-driven decisions with statistical rigor

---

## Slide 33: Why This Matters - Educational Value
**Teaching Modern MLOps**

- Complete end-to-end workflow demonstrated
- Industry-standard patterns (Champion/Challenger)
- Two frameworks (TensorFlow + PyTorch)

---

## Slide 34: Why This Matters - Scalability
**Built to Grow**

- Environment-based naming (dev → pilot → prod)
- Infrastructure as Code (reproducible)
- Terraform state management

---

## Slide 35: Technical Stack
**Modern, Industry-Standard Tools**

- **Cloud**: AWS (S3, IAM)
- **Platform**: Databricks on AWS
- **Governance**: Unity Catalog

---

## Slide 36: Technical Stack (cont.)
**MLOps & Storage**

- **Storage**: Delta Lake (ACID transactions)
- **ML**: MLflow + TensorFlow + PyTorch
- **Serving**: Databricks Model Serving

---

## Slide 37: Technical Stack (cont.)
**Deployment & Monitoring**

- **IaC**: Terraform (everything versioned)
- **Tracking**: GitHub Issues
- **Monitoring**: Custom feedback dashboard

---

## Slide 38: Key Achievements - Infrastructure
**Solid Foundation**

- ✅ AWS S3 integration with Unity Catalog
- ✅ Production Delta tables with Terraform
- ✅ 1000 X-ray images ingested

---

## Slide 39: Key Achievements - ML Pipeline
**End-to-End Workflow**

- ✅ Two trained models (Keras + PyTorch)
- ✅ MLflow registry with Unity Catalog
- ✅ Model serving endpoints deployed

---

## Slide 40: Key Achievements - A/B Testing
**Production MLOps**

- ✅ Champion/Challenger A/B testing infrastructure
- ✅ Feedback loop with ground truth
- ✅ Monitoring dashboard with statistical testing

---

## Slide 41: What Makes This Special - Completeness
**Every Piece of Production MLOps**

- Most demos stop at training
- This includes deployment, monitoring, improvement
- Real feedback loop with ground truth

---

## Slide 42: What Makes This Special - Best Practices
**Industry-Standard Patterns**

- Champion/Challenger (not just model replacement)
- Statistical significance testing (not gut feelings)
- Infrastructure as Code (reproducible)

---

## Slide 43: What Makes This Special - Governance
**Enterprise-Ready**

- Unity Catalog for centralized governance
- Delta Lake for ACID guarantees
- Terraform for audit trail

---

## Slide 44: The Meta-Story - AI-Assisted Development
**What Really Makes This Special**

- Not just the healthcare platform
- But HOW it was built
- Human + AI collaboration demonstrated

---

## Slide 45: Claude Code - The Development Partner
**Running Locally with Full Access**

- GitHub CLI: Creates issues, manages tickets
- AWS CLI: Deploys S3, IAM, verifies infrastructure
- Databricks CLI: Uploads notebooks, manages clusters

---

## Slide 46: Beyond Code Generation
**AI Handled Every Phase**

- Planning: Created all 15 GitHub issues
- Infrastructure: Wrote Terraform, ran deployments
- Development: 9 notebooks with educational comments

---

## Slide 47: Beyond Code Generation (cont.)
**Complete Development Lifecycle**

- Testing: Debugged errors, created bug issues
- Documentation: README + 71-slide presentation
- Deployment: Actual AWS/Databricks deployments executed

---

## Slide 48: The Rubber Duck Effect
**AI as Thinking Partner**

- "What is Champion/Challenger?" → Detailed explanation
- "Why TensorFlow not PyTorch?" → Trade-off discussion
- "How to handle feedback?" → Architecture proposal

---

## Slide 49: Terraform's Dual Purpose
**IaC AND AI Context**

- Traditional: Infrastructure as Code
- Novel: Perfect AI context builder
- AI reads schemas, writes correct code first try

---

## Slide 50: The Numbers - AI Contribution
**Measurable Impact**

- 15 GitHub issues: 100% AI-created
- ~1200 lines Terraform: 95% AI-written
- ~2000 lines notebooks: 90% AI-written

---

## Slide 51: The Numbers - AI Contribution (cont.)
**Execution Speed**

- Documentation: 100% AI-written
- Deployments: AI-executed to cloud
- Time to MVP: 3-4 days (10x faster)

---

## Slide 52: Value for Staff Engineers
**Why Experienced Engineers Benefit**

- Speed: Days not weeks
- Context switching: AI handles multiple systems
- Documentation: No burden left behind

---

## Slide 53: Value for Staff Engineers (cont.)
**Quality Improvements**

- Rubber duck: Architecture discussion partner
- Learning: AI explains unfamiliar features
- Best practices: Applied automatically

---

## Slide 54: The Collaboration Pattern
**Division of Labor**

- Human: Vision, approval, strategy, business context
- AI: Issues, code, deploy, test, document
- Result: Staff engineer becomes architect

---

## Slide 55: Traditional vs AI-Assisted
**The Old Way**

- Write code locally
- Copy-paste to cloud console
- Manually upload files

---

## Slide 56: Traditional vs AI-Assisted (cont.)
**The New Way**

- Describe what you want
- AI creates issue, writes, deploys, documents
- Everything committed and linked automatically

---

## Slide 57: Reproducibility Guarantee
**Because AI Has Direct System Access**

- No copy-paste errors
- No manual uploads
- Everything in Git history

---

## Slide 58: The Future of Development
**Shift from Typing to Architecting**

- AI handles tactical execution
- Human provides strategic direction
- Documentation happens automatically

---

## Slide 59: Lessons Learned - Cold Starts
**Serverless Endpoints Need Warm-Up**

- First request: 30-60 seconds
- Solution: Dedicated warm-up request
- Trade-off: Cost vs latency

---

## Slide 45: Lessons Learned - Model Signatures
**Unity Catalog Requires Schemas**

- Must define input/output schema
- Use `infer_signature()` during training
- Enables validation and compatibility checks

---

## Slide 46: Lessons Learned - Payload Formats
**Different Frameworks, Different Expectations**

- TensorFlow/Keras: `{"inputs": [array]}`
- PyTorch: Varies by wrapper
- Always test with sample requests

---

## Slide 47: Future Enhancements - Better Models
**From POC to Production Quality**

- Transfer learning (EfficientNet, ResNet)
- Larger dataset (full 5000+ images)
- Hyperparameter tuning with AutoML

---

## Slide 48: Future Enhancements - Feature Engineering
**Improving Data Quality**

- Bronze → Silver transformation pipeline
- Data quality checks and validation
- Feature extraction and engineering

---

## Slide 49: Future Enhancements - Automation
**CI/CD and Auto-Retraining**

- GitHub Actions for deployment
- Automated retraining on feedback
- Real-time alerting for model drift

---

## Slide 50: The Journey Recap
**What We Built**

- From AWS buckets to production AI
- From raw images to actionable insights
- From single models to A/B testing

---

## Slide 51: The Journey Recap (cont.)
**How We Built It**

- Infrastructure as Code (Terraform)
- Issue-driven development (GitHub)
- Best practices throughout

---

## Slide 52: The Business Value
**Why This Matters to Healthcare**

- Safer AI deployment (A/B testing)
- Data-driven decisions (feedback loop)
- Continuous improvement (challenger promotion)

---

## Slide 53: The Educational Value
**Why This Matters to Students**

- Complete MLOps workflow demonstrated
- Both TensorFlow and PyTorch
- Real-world production patterns

---

## Slide 54: The Technical Value
**Why This Matters to Engineers**

- Reproducible infrastructure (Terraform)
- Governed assets (Unity Catalog)
- Scalable architecture (Databricks + AWS)

---

## Slide 55: Try It Yourself - Setup
**Getting Started**

1. Clone repository from GitHub
2. Configure AWS CLI and Databricks
3. Run `terraform apply` (5-7 minutes)

---

## Slide 56: Try It Yourself - Ingest Data
**Load the X-Ray Images**

1. Open `/Shared/ingest-kaggle-xray-data` notebook
2. Add Kaggle credentials to secrets
3. Run notebook (10-15 minutes)

---

## Slide 57: Try It Yourself - Train Models
**Create the Classifiers**

1. Run `/Shared/train-poc-model` (TensorFlow)
2. Run `/Shared/train-poc-model-pytorch` (PyTorch)
3. Check MLflow registry

---

## Slide 58: Try It Yourself - Deploy A/B Endpoint
**Set Up Champion vs Challenger**

1. Run `/Shared/deploy-ab-testing-endpoint` notebook
2. Wait 5-10 minutes for endpoint
3. Verify in Model Serving UI

---

## Slide 59: Try It Yourself - Make Predictions
**Test the System**

1. Run `/Shared/demo-model-usage` notebook
2. See predictions from both models
3. Capture request IDs

---

## Slide 60: Try It Yourself - Submit Feedback
**Close the Loop**

1. Use `submit_feedback()` function
2. Simulate radiologist ground truth
3. Query `model_performance_live` view

---

## Slide 61: Try It Yourself - Monitor Performance
**Compare Champion vs Challenger**

1. Run `/Shared/monitor-ab-test` notebook
2. View traffic distribution charts
3. See accuracy comparison and recommendations

---

## Slide 62: Resources - Documentation
**Where to Learn More**

- GitHub repository with all code
- 15 issues documenting every step
- README with detailed setup guide

---

## Slide 63: Resources - Notebooks
**Hands-On Learning**

- 9 Databricks notebooks included
- Commented code with explanations
- Educational notes throughout

---

## Slide 64: Resources - Community
**Built with Claude Code**

- Infrastructure managed with AI assistance
- Issue-driven development approach
- Every commit linked to issues

---

## Slide 80: Final Thoughts - Dual Value Proposition
**Two Stories in One**

- Story 1: Production-ready healthcare AI platform
- Story 2: AI-assisted development workflow
- Both equally valuable and demonstrable

---

## Slide 81: Final Thoughts - The Technical Achievement
**Complete MLOps Platform**

- End-to-end workflow: Train → Monitor → Improve
- A/B testing for safe updates
- Feedback loop for continuous improvement

---

## Slide 82: Final Thoughts - The Process Achievement
**AI-Accelerated Development**

- Claude Code with direct system access
- 10x faster: 3-4 days to MVP
- Staff engineer as architect, AI as executor

---

## Slide 83: The Real Impact - For Healthcare
**Production-Ready AI**

- Doctors get reliable predictions
- Models continuously improve
- Platform scales to real hospitals

---

## Slide 84: The Real Impact - For Engineers
**Better Development Experience**

- Focus on architecture, not typing
- Documentation happens automatically
- Rubber duck partner always available

---

## Slide 85: Call to Action - For Students
**Learn Modern MLOps**

- Run the notebooks yourself
- Experiment with different models
- Understand production patterns

---

## Slide 86: Call to Action - For Engineers
**Build Better ML Systems**

- Use Champion/Challenger pattern
- Implement feedback loops
- Track real-world accuracy

---

## Slide 87: Call to Action - Try Claude Code
**Experience AI-Assisted Development**

- Install Claude Code CLI locally
- Give it access to GitHub/AWS/Databricks
- Watch it plan, build, deploy, document

---

## Slide 88: Call to Action - For Teams
**Deploy This to Production**

- Clone repository, customize for use case
- Add your own models and data
- Benefit from proven architecture and workflow

---

## Slide 89: Thank You
**Questions?**

- Repository: github.com/rgasiorek/healthcare-ai-platform-databricks
- All 15 issues with detailed documentation
- Built with Claude Code + staff engineer collaboration

---

## Slide 90: Key Takeaways
**Remember These Three Things**

1. Complete MLOps: Train → Monitor → Improve
2. A/B Testing: Safe model updates matter
3. AI-Assisted Development: The future is here

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
