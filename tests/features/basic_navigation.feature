Feature: Basic Navigation
  As a user
  I want to navigate through the TV interface
  So that I can access different functions

  Background:
    Given the test platform is initialized
    And the device is powered on

  Scenario: Navigate to home screen
    Given the device is on any screen
    When I press the HOME button
    Then I should see the home screen
    And no anomalies should be present

  Scenario: Navigate using direction keys
    Given the device is on home screen
    When I press RIGHT arrow
    Then the focus should move to the right
    When I press DOWN arrow  
    Then the focus should move down
    When I press LEFT arrow
    Then the focus should move to the left
    When I press UP arrow
    Then the focus should move up

  Scenario: Go back from any screen
    Given the device is on home screen
    When I navigate to any menu
    And I press BACK button
    Then I should return to the previous screen