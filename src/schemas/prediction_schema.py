from pydantic import BaseModel, Field
from typing import Annotated, Literal
from pydantic import BaseModel


# ---------------- INPUT SCHEMA ---------------- #
# class LoanInput(BaseModel):
#     Age: Annotated[int, Field(..., description="Age of the applicant", gt=0, lt=100, examples=[25, 40, 60])]
#     Income: Annotated[float, Field(..., description="Annual income of the applicant", gt=0, examples=[30000, 75000, 150000])]
#     LoanAmount: Annotated[float, Field(..., description="Requested loan amount", gt=0)]
#     CreditScore: Annotated[int, Field(..., description="Credit score of the applicant", ge=300, le=850, examples=[301, 850])]
#     MonthsEmployed: Annotated[int, Field(..., description="Number of months employed", ge=0)]
#     NumCreditLines: Annotated[int, Field(..., description="Number of credit lines", ge=0, examples=[0, 5, 10])]
#     InterestRate: Annotated[float, Field(..., description="Interest rate of the loan", gt=0, lt=100, examples=[5.5, 12.5, 25.0])]
#     LoanTerm: Annotated[int, Field(..., description="Term of the loan in months", examples=[12, 36, 60])]
#     DTIRatio: Annotated[float, Field(..., description="Debt-to-income ratio", gt=0, lt=1, examples=[0.1, 0.3, 0.5])]
#     Education: Annotated[str, Field(..., description="Education level of the applicant", examples=["High School", "Bachelor's", "Master's", "PhD"])]
#     EmploymentType: Annotated[str, Field(..., description="Type of employment", examples=["Full-time", "Part-time", "Unemployed"])]
#     MaritalStatus: Annotated[str, Field(..., description="Marital status of the applicant", examples=["Single", "Married", "Divorced"])]
#     HasMortgage: Annotated[str, Field(..., description="Whether the applicant has a mortgage", examples=["Yes", "No"])]
#     HasDependents: Annotated[str, Field(..., description="Whether the applicant has dependents", examples=["Yes", "No"])]
#     LoanPurpose: Annotated[str, Field(..., description="Purpose of the loan", examples=["Business", "Home", "Education", "Other","Auto"])]
#     HasCoSigner: Annotated[str, Field(..., description="Whether the loan has a co-signer", examples=["Yes", "No"])]




# ---------------- INPUT SCHEMA ---------------- #
class LoanInput(BaseModel):

    Age: Annotated[
        int,
        Field(gt=0, lt=100, description="Age of applicant")
    ]

    Income: Annotated[
        float,
        Field(gt=0, description="Annual income")
    ]

    LoanAmount: Annotated[
        float,
        Field(gt=0, description="Loan amount requested")
    ]

    CreditScore: Annotated[
        int,
        Field(ge=300, le=850, description="Credit score")
    ]

    MonthsEmployed: Annotated[
        int,
        Field(ge=0)
    ]

    NumCreditLines: Annotated[
        int,
        Field(ge=0)
    ]

    InterestRate: Annotated[
        float,
        Field(gt=0, lt=100)
    ]

    LoanTerm: Annotated[
        int,
        Field(gt=0)
    ]

    DTIRatio: Annotated[
        float,
        Field(ge=0, le=1)
    ]

    Education: Literal[
        "High School",
        "Bachelor's",
        "Master's",
        "PhD"
    ]

    EmploymentType: Literal[
        "Full-time",
        "Part-time",
        "Unemployed",
        "Self-employed"
    ]

    MaritalStatus: Literal[
        "Single",
        "Married",
        "Divorced"
    ]

    HasMortgage: Literal["Yes", "No"]

    HasDependents: Literal["Yes", "No"]

    LoanPurpose: Literal[
        "Business",
        "Home",
        "Education",
        "Other",
        "Auto"
    ]

    HasCoSigner: Literal["Yes", "No"]


# ---------------- RESPONSE SCHEMA ---------------- #
class PredictionResponse(BaseModel):

    prediction: int = Field(
        description="0 = No Default, 1 = Default"
    )

    result: str = Field(
        description="Human-readable prediction"
    )