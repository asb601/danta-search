// ============================================================================
// vmss.bicep — danta-search self-tuning ingestion worker fleet
// ----------------------------------------------------------------------------
// One identical worker image, scaled horizontally by Azure Monitor autoscale
// (see autoscale.sh). ZERO per-node tuning: on boot, resource_profile.py reads
// cgroup/sysconf and compute_ingestion_knobs() sizes every concurrency knob, so
// a D4s_v5 and a D16s_v5 "just work" from the SAME image — that is why the
// worker launch command in cloud-init.yaml carries NO -c concurrency flag.
//
// Deploy (Section 5.5 step 3):
//   az deployment group create -g rg-danta-ingest --template-file vmss.bicep \
//     --parameters redisUrl="rediss://:pwd@redis-gchat:6380/0"
// ============================================================================

@description('Azure region. Defaults to the resource group location.')
param location string = resourceGroup().location

@description('VM SKU for each worker. Start at D4s_v5; bump to D16s_v5 freely — the worker auto-tunes to the box it lands on.')
param sku string = 'Standard_D4s_v5'

@description('Capacity floor. Keep at 1 so the broker stays reachable and the queue-depth metric publisher (systemd timer on instance 0) stays alive. Set 0 only if you accept cold-start lag on the first burst.')
@minValue(0)
param capacity int = 1

@description('Celery broker URL (Azure Cache for Redis, db0). Format: rediss://:PASSWORD@HOST:6380/0')
@secure()
param redisUrl string

@description('Admin username for the Linux worker VMs (key-based auth only).')
param adminUser string = 'gchat'

@description('SSH public key for the admin user (password auth is disabled).')
param adminSshPublicKey string

@description('Existing subnet resource ID the scale set NICs attach to.')
param subnetId string

// cloud-init.yaml is the entire per-node "config": installs uv, pulls the repo,
// drops the systemd unit that launches the self-tuned CPU-lane worker, and
// enables the queue-depth metric timer. Templated so ${REDIS_URL} is injected.
var cloudInit = replace(loadTextContent('cloud-init.yaml'), '__REDIS_URL__', redisUrl)

resource vmss 'Microsoft.Compute/virtualMachineScaleSets@2024-03-01' = {
  name: 'vmss-gchat-worker'
  location: location
  sku: {
    name: sku
    tier: 'Standard'
    capacity: capacity // floor; autoscale rules drive the live count 1..40
  }
  // SystemAssigned identity is what lets publish_queue_depth.py POST the custom
  // metric to Azure Monitor without a stored secret (ManagedIdentityCredential).
  // Grant it "Monitoring Metrics Publisher" on this VMSS (Section 5.5 step 4).
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    // Overprovision OFF: we do not want transient extra VMs racing for tasks;
    // each VM is a heavy, RAM-bound worker, not a stateless web node.
    overprovision: false
    // Rolling upgrade so a bad image is replaced gradually. Combined with the
    // worker's task_acks_late + task_reject_on_worker_lost (already set in
    // celery_app.py), a VM cycled mid-task returns its in-flight task to Redis.
    upgradePolicy: {
      mode: 'Rolling'
      rollingUpgradePolicy: {
        maxBatchInstancePercent: 20
        maxUnhealthyInstancePercent: 20
        maxUnhealthyUpgradedInstancePercent: 20
        pauseTimeBetweenBatches: 'PT2M'
      }
    }
    virtualMachineProfile: {
      osProfile: {
        computerNamePrefix: 'gchatwrk'
        adminUsername: adminUser
        // customData (cloud-init) is base64-encoded by ARM at deploy time.
        customData: base64(cloudInit)
        linuxConfiguration: {
          disablePasswordAuthentication: true
          ssh: {
            publicKeys: [
              {
                path: '/home/${adminUser}/.ssh/authorized_keys'
                keyData: adminSshPublicKey
              }
            ]
          }
        }
      }
      storageProfile: {
        imageReference: {
          publisher: 'Canonical'
          offer: 'ubuntu-24_04-lts'
          sku: 'server'
          version: 'latest'
        }
        osDisk: {
          createOption: 'FromImage'
          caching: 'ReadWrite'
          managedDisk: {
            storageAccountType: 'Premium_LRS'
          }
        }
      }
      networkProfile: {
        networkInterfaceConfigurations: [
          {
            name: 'nic-gchat-worker'
            properties: {
              primary: true
              ipConfigurations: [
                {
                  name: 'ipconfig-gchat-worker'
                  properties: {
                    subnet: {
                      id: subnetId
                    }
                  }
                }
              ]
            }
          }
        ]
      }
    }
  }
}

@description('Resource ID of the scale set (feed to autoscale.sh as RES_ID and to publish_queue_depth.py as VMSS_RESOURCE_ID).')
output vmssId string = vmss.id

@description('Principal ID of the system-assigned identity. Grant it "Monitoring Metrics Publisher" on the VMSS.')
output vmssIdentityPrincipalId string = vmss.identity.principalId
