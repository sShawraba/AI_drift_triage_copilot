# 1. Health check
Invoke-RestMethod -Uri http://localhost:8001/health

# 2. Send a drift event (should pause for approval)
$body = @{
    event_id      = "evt-docker-test"
    timestamp     = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    model_version = "v1.0"
    psi_numeric   = 0.85
    chi2_categorical = 0.72
    output_drift  = 0.91
    severity      = "high"
} | ConvertTo-Json

$response = Invoke-RestMethod -Uri http://localhost:8001/webhook/drift-event -Method Post -Body $body -ContentType "application/json"
$response
$invId = $response.investigation_id

# 3. List pending approvals
Invoke-RestMethod -Uri http://localhost:8001/webhook/pending

# 4. Approve the investigation
Invoke-RestMethod -Uri "http://localhost:8001/webhook/investigations/$invId/approve" -Method Post -Body '{"approve":true}' -ContentType "application/json"

# 5. Observe the worker logs (should show the job processed)
docker compose logs worker

# 6. Verify investigation status in the database
docker exec -it $(docker ps -qf "name=postgres") psql -U admin -d mlops -c "SELECT id, status, proposed_action, final_decision FROM investigations WHERE id='$invId';"

# 7. Queue stats
Invoke-RestMethod -Uri http://localhost:8001/webhook/queue/stats