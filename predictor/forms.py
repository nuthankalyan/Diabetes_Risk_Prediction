from django import forms

class DiabetesPredictionForm(forms.Form):
    Age = forms.IntegerField(min_value=0, max_value=120)
    Pregnancies = forms.IntegerField(min_value=0, max_value=20)
    BMI = forms.FloatField(min_value=10, max_value=50)
    Glucose = forms.FloatField(min_value=50, max_value=200)
    BloodPressure = forms.FloatField(min_value=50, max_value=200)
    HbA1c = forms.FloatField(min_value=3, max_value=9)
    LDL = forms.FloatField(min_value=30, max_value=200)
    HDL = forms.FloatField(min_value=15, max_value=100)
    Triglycerides = forms.FloatField(min_value=50, max_value=300)
    WaistCircumference = forms.FloatField(min_value=50, max_value=150)
    HipCircumference = forms.FloatField(min_value=50, max_value=150)
    WHR = forms.FloatField(min_value=0.5, max_value=2)
    FamilyHistory = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')])
    DietType = forms.ChoiceField(choices=[(0, 'Regular'), (1, 'Vegetarian'), (2, 'Other')])
    Hypertension = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')])
    MedicationUse = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields:
            self.fields[field].widget.attrs.update({'class': 'form-control'})
            self.fields[field].required = True 