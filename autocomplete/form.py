from django import forms

class AutoCompleteForm(forms.Form):
    document = forms.CharField(label=" ", widget=forms.Textarea())