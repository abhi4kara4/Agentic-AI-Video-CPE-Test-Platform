Feature: Netflix App Launch
  As a user
  I want to launch Netflix from the app rail
  So that I can watch content

  Background:
    Given the test platform is initialized
    And the device is powered on

  Scenario: Launch Netflix from App Rail
    Given the device is on home screen
    When I navigate to Netflix in app rail
    And I press OK
    Then Netflix should launch
    And I should see either login screen or profile selection or home screen
    And I should not see black screen

  Scenario: Launch Netflix and verify no anomalies
    Given the device is on home screen
    When I launch the Netflix app
    Then the app should load without anomalies
    And no buffering indicator should be present
    And the screen should not be frozen