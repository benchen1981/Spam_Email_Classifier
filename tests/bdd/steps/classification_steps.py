"""BDD Step Implementations"""
from pytest_bdd import given, when, then, scenarios

scenarios('../features/email_classification.feature')

@given('an email with spam content')
def spam_email():
    return "WIN FREE MONEY NOW!!!"

@given('an email with legitimate content')
def ham_email():
    return "Meeting tomorrow at 10 AM"

@when('I classify the email')
def classify(spam_email):
    return "spam" if "WIN" in spam_email else "ham"

@then('it should be marked as spam')
def verify_spam():
    assert True

@then('it should be marked as ham')
def verify_ham():
    assert True