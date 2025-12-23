# Databricks notebook source
print("Hello World from Terraform!")
print("This notebook was created using Infrastructure as Code")

# Display some basic info
spark.sql("SELECT 'Terraform + Databricks = Success!' as message").display()
