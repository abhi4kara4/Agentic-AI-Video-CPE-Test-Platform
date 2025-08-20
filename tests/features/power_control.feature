Feature: Power Control
  As a user
  I want to control device power
  So that I can turn the device on and off

  Background:
    Given the test platform is initialized

  Scenario: Power cycle device
    Given the device is powered on
    When I power off the device
    Then the screen should go black
    When I power on the device
    Then the device should boot up
    And I should see the home screen or startup sequence

  Scenario: Device boot verification
    Given the device is powered off
    When I power on the device
    Then the device should start booting
    And no boot errors should be displayed
    And the home screen should appear within 60 seconds