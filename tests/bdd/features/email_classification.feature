Feature: Email Classification
  As a user
  I want to classify emails
  So that I can identify spam

  Scenario: Classify spam email
    Given an email with spam content
    When I classify the email
    Then it should be marked as spam
    
  Scenario: Classify legitimate email
    Given an email with legitimate content
    When I classify the email
    Then it should be marked as ham