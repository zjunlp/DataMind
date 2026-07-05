from pydantic import BaseModel, Field

class SkillModelOutput(BaseModel):
    """The schema exposed to the model through StructuredOutput."""
    reasoning: str
    generated_skill: str