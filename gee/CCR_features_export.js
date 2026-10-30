// ============================================================
// MULTI-SITE CCR FEATURE EXPORT (features only)
// Sites: Glen Lyn · Bremo · Chesterfield · Possum Point
// ─────────────────────────────────────────────────────────────
// Builds the main feature stack (2019-2025 summer medians, Cloud Score+
// masking, scene filter SCENE_CLOUD_MAX) and one stack per year, each with
// the 66 bands (names and order) of the original stack, prints the band
// names, exports the stacks to Drive (Tasks tab) and adds every band,
// clipped to the ROI, to the map (MAP DISPLAY SETTINGS). Labels and kriged
// HI only with EXPORT_LABELS = true.
// TO SWITCH SITES: change selectedSite to one of
//   'Glen Lyn' | 'Bremo' | 'Chesterfield' | 'Possum Point'
// ============================================================
var selectedSite = 'Glen Lyn';
 
// ========================== SITE CONFIG =============================
var siteConfigs = {
  'Glen Lyn': {
    roi: ee.Geometry.Polygon([[
      [-80.88602532062332, 37.36408912138987],
      [-80.86027611408035, 37.36408912138987],
      [-80.86027611408035, 37.38305127717347],
      [-80.88602532062332, 37.38305127717347],
      [-80.88602532062332, 37.36408912138987]
    ]]),
    asset:   'projects/ee-salmankarim/assets/GlenLyn_wells_HI_composite',
    kriging: { range: 1500, sill: 50,  nugget: 3,  maxDistance: 2500 },
    zoom:    14,
    rawMax:  25,
    splitHI: 5.0   // used only with LABEL_RULE = 'HI'
  },
  'Bremo': {
    roi: ee.Geometry.Polygon([[
      [-78.30039054301348, 37.69660048434903],
      [-78.27352553752031, 37.69660048434903],
      [-78.27352553752031, 37.71500299536265],
      [-78.30039054301348, 37.71500299536265],
      [-78.30039054301348, 37.69660048434903]
    ]]),
    asset:   'projects/ee-salmankarim/assets/Bremo_wells_HI_composite',
    kriging: { range: 2000, sill: 75,  nugget: 5,  maxDistance: 3000 },
    zoom:    14,
    rawMax:  35,
    splitHI: 5.0   // used only with LABEL_RULE = 'HI'
  },
  'Chesterfield': {
    roi: ee.Geometry.Polygon([[
      [-77.41016043242651, 37.36006733962347],
      [-77.3585761886521,  37.36006733962347],
      [-77.3585761886521,  37.39553411886849],
      [-77.41016043242651, 37.39553411886849],
      [-77.41016043242651, 37.36006733962347]
    ]]),
    asset:   'projects/ee-salmankarim/assets/Chesterfield_wells_HI_composite',
    kriging: { range: 2500, sill: 200, nugget: 10, maxDistance: 4000 },
    zoom:    13,
    rawMax:  60,
    splitHI: 5
  },
  'Possum Point': {
    roi: ee.Geometry.Polygon([[
      [-77.30699481987425, 38.53014484806756],
      [-77.26262035393187, 38.53014484806756],
      [-77.26262035393187, 38.565386642500634],
      [-77.30699481987425, 38.565386642500634],
      [-77.30699481987425, 38.53014484806756]
    ]]),
    asset:   'projects/ee-salmankarim/assets/PossumPoint_wells_HI_composite',
    kriging: { range: 1800, sill: 30,  nugget: 2,  maxDistance: 2800 },
    zoom:    13,
    rawMax:  20,
    splitHI: 5
  }
};
 
// ── Resolve config for selected site ──
var cfg = siteConfigs[selectedSite];
var roi = cfg.roi;
 
 
Map.centerObject(roi, cfg.zoom);
Map.setOptions('HYBRID');
Map.addLayer(roi, { color: 'FFFFFF', fillColor: '00000000', width: 2 },
  '📦 ROI — ' + selectedSite);
print('▶ Site:', selectedSite);
// ---- binary label: alpha split on the normalised kriged HI ------------------------------------------
var LABEL_RULE = 'percentile';   // 'percentile': alpha = the ALPHA_PERCENTILE-th percentile of HI_norm, so the
                                 //   lowest 25% of the grid is class 0 and the rest class 1 (HazardFuse labels);
                                 // 'HI': alpha = normalised position of cfg.splitHI (class 1 = kriged HI >= splitHI)
var ALPHA_PERCENTILE = 25;       // percent of the grid in class 0 with LABEL_RULE = 'percentile'
var LABEL_CRS = 'EPSG:4326', LABEL_SCALE = 10;   // grid of the exported label (as glen_lyn_Label.tif)

function makeLabel(krigedHI, region, splitHI) {
  // min-max normalisation and alpha on the label grid; percentile exact up to 1e6 pixels (maxRaw)
  function args(reducer) {
    return {reducer: reducer, geometry: region, crs: LABEL_CRS, scale: LABEL_SCALE, maxPixels: 1e13, tileScale: 4};
  }
  var st = krigedHI.reduceRegion(args(ee.Reducer.minMax()));
  var hiMin = ee.Number(st.get('HI_kriged_min')), hiMax = ee.Number(st.get('HI_kriged_max'));
  var hiNorm = krigedHI.subtract(hiMin).divide(hiMax.subtract(hiMin).max(1e-12)).rename('HI_norm');
  var alpha = LABEL_RULE === 'percentile'
    ? ee.Number(hiNorm.reduceRegion(args(
        ee.Reducer.percentile({percentiles: [ALPHA_PERCENTILE], maxRaw: 1e6}))).values().get(0))
    : ee.Number(splitHI).subtract(hiMin).divide(hiMax.subtract(hiMin).max(1e-12));
  var label = hiNorm.gte(alpha).toByte().rename('class');
  return {hiMin: hiMin, hiMax: hiMax, hiNorm: hiNorm, alpha: alpha, label: label,
          hiAtAlpha: hiMin.add(alpha.multiply(hiMax.subtract(hiMin)))};
}
 
// =====================================================================
// FEATURE SETTINGS: EDIT here
// ---------------------------------------------------------------------
// Composites are summer (June-August) medians: one multi-year composite
// for the main stack and one per year for the yearly stacks. Clouds and
// shadows are masked with Cloud Score+ (QA60 is incomplete from 2022 on).
// =====================================================================
var START_YEAR = 2019, END_YEAR = 2025;
var YEARS = [];
for (var yy = START_YEAR; yy <= END_YEAR; yy++) { YEARS.push(yy); }
var SUMMER = [6, 8];                 // months of the summer composite (June-August)
var SPRING = [4, 5], AUTUMN = [9, 10];   // seasonal features (SEASONAL_FEATURES)
var CLOUD_MASK = 'csplus';           // 'csplus' (Cloud Score+) or 'qa60' (previous behaviour)
var CS_THRESHOLD = 0.6;              // Cloud Score+ cs_cdf: keep pixels >= 0.6 (0.5-0.65 usual)
var SCENE_CLOUD_MAX = 30;            // scene filter (%); per-pixel masking does the rest
var INDEX_CLAMP = {MTCI: [-2, 15], CIRE: [-1, 20], LCC: [0, 5], CCC: [-2, 15]};   // ratio indices, per image
var RATE_METHOD = 'sens';            // 'sens' (Sen's slope, robust) or 'ols' (least squares); per year
var CSTV_NDVI_MIN = 0.2;             // CSTV only where summer NDVI of START_YEAR >= this
var CSTV_FILL = 0;                   // value of CSTV where masked (0 = no change)
var ADD_CSTV_MASK = false;           // add band CSTV_valid (only with STACK_LAYOUT = 'extended')
var STATIC_CLIMATE = 'coarse';      // main stack: 'landsat' (Temperature = Landsat summer LST median, 100 m
                                     //   thermal; coarse humidity, soil moisture, precipitation dropped),
                                     //   'coarse' (previous MODIS/NLDAS/GLDAS/CHIRPS bands) or 'none'
var YEARLY_FORCING = true;           // yearly stacks: Temperature, Humidity, PrecipJJA_mm_mean of that summer
var SEASONAL_FEATURES = 'none';      // 'none', 'seasonal' (spring/autumn NDVI and their difference to summer),
                                     //   'harmonic' (amplitude and phase of the annual NDVI cycle) or 'both';
                                     //   test against 'none' on the validation blocks before keeping it
var S2_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'];
var VI_NAMES = ['NDVI', 'MTCI', 'SAVI', 'CIRE', 'NDRE1', 'NDRE2', 'LAI', 'LCC', 'CCC'];
var DW_SRC = ['water', 'trees', 'grass', 'flooded_vegetation', 'crops', 'shrub_and_scrub', 'built', 'bare',
              'snow_and_ice'];
var DW_NAMES = ['DW_Water', 'DW_Trees', 'DW_Grass', 'DW_FloodedVeg', 'DW_Crops', 'DW_Shrubs', 'DW_Built',
                'DW_Bare', 'DW_Snow'];
// band layout of the exported stacks
var STACK_LAYOUT = 'hazardfuse66';   // 'hazardfuse66': the 66 bands, names and order of the original stack, in the
                                     //   main AND every yearly stack (HazardFuse, HazardAE and HazardSSL read them
                                     //   unchanged); 'extended': every band built (CSTV_valid, seasonal, ...)
var BANDS_66 = ['NDVI', 'MTCI', 'SAVI', 'CIRE', 'NDRE1', 'NDRE2', 'LAI', 'LCC', 'CCC',
  'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'QA60',
  'CSTV_NDVI_Diff', 'CSTV_MTCI_Diff', 'CSTV_SAVI_Diff', 'CSTV_CIRE_Diff', 'CSTV_NDRE1_Diff', 'CSTV_NDRE2_Diff',
  'CSTV_LAI_Diff', 'CSTV_LCC_Diff', 'CSTV_CCC_Diff',
  'DEM', 'Slope_deg', 'Aspect_deg', 'Curvature', 'Temperature', 'Humidity', 'Moisture',
  'pH', 'Sand_pct_0_5cm', 'Silt_pct_0_5cm', 'Clay_pct_0_5cm', 'PrecipJJA_mm_mean', 'UpstreamArea_km2', 'HAND_m', 'TWI',
  'NDVI_Rate', 'MTCI_Rate', 'SAVI_Rate', 'CIRE_Rate', 'NDRE1_Rate', 'NDRE2_Rate', 'LAI_Rate', 'LCC_Rate', 'CCC_Rate',
  'DW_Water', 'DW_Trees', 'DW_Grass', 'DW_FloodedVeg', 'DW_Crops', 'DW_Shrubs', 'DW_Built', 'DW_Bare', 'DW_Snow',
  'DW_LandCover', 'DW_Vegetation'];
// yearly CSTV and rates: 'to_date' = change from START_YEAR to that year and Sen's slope over START_YEAR..that year
// (no information from later years; the last year equals the main stack; the first year is 0);
// 'full' = the 2019-2025 values of the main stack repeated in every year
var YEARLY_CHANGE = 'to_date';
if (STACK_LAYOUT === 'hazardfuse66') STATIC_CLIMATE = 'coarse';   // the 66 bands need Temperature, Humidity,
                                                                 // Moisture and PrecipJJA_mm_mean (MODIS/NLDAS/GLDAS/CHIRPS)
// exports (grid of the HazardFuse / HazardSSL rasters: EPSG:4326, 10 m)
var EXPORT_MAIN = true;              // <site>_Features
var EXPORT_YEARLY = true;            // <site>_FeatureStack_<year> for every year
var EXPORT_LABELS = true;           // true: also <site>_Label, <site>_HI and <site>_Label_<year> (kriged HI)
var EXPORT_FOLDER = 'HazardSSL_Yearly_Feature_Stacks';
var EXPORT_CRS = 'EPSG:4326', EXPORT_SCALE = 10;
// optional per-year well composites ({2019: 'projects/.../GlenLyn_wells_HI_2019', ...}); years not listed use
// the site composite, so their label is the same raster every year
var YEARLY_LABEL_ASSETS = {};

// =====================================================================
// FEATURES
// =====================================================================
function s2Collection(region, cloudMax) {               // cloudMax: scene filter (%), default SCENE_CLOUD_MAX
  var col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(region)
    .filterDate(ee.Date.fromYMD(START_YEAR, 1, 1), ee.Date.fromYMD(END_YEAR + 1, 1, 1))
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloudMax === undefined ? SCENE_CLOUD_MAX : cloudMax));
  if (CLOUD_MASK === 'csplus') {
    col = col.linkCollection(ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED'), ['cs_cdf'])
      .map(function(img) {
        var ok = img.select('cs_cdf').gte(CS_THRESHOLD);
        return img.select(S2_BANDS).updateMask(ok).divide(10000)
          .addBands(img.select('QA60').updateMask(ok).toFloat())   // QA60 kept for the 66-band layout
          .copyProperties(img, ['system:time_start']);
      });
  } else {
    col = col.map(function(img) {
      var qa = img.select('QA60');
      var ok = qa.bitwiseAnd(1 << 10).eq(0).and(qa.bitwiseAnd(1 << 11).eq(0));
      return img.select(S2_BANDS).updateMask(ok).divide(10000).addBands(qa.updateMask(ok).toFloat())
        .copyProperties(img, ['system:time_start']);
    });
  }
  return col;
}

function addIndices(image) {
  var b = function(n) { return image.select(n); };
  var idx = [
    image.normalizedDifference(['B8', 'B4']).rename('NDVI'),
    b('B6').subtract(b('B5')).divide(b('B5').subtract(b('B4'))).rename('MTCI'),
    b('B8').subtract(b('B4')).divide(b('B8').add(b('B4')).add(0.5)).multiply(1.5).rename('SAVI'),
    b('B7').divide(b('B5')).subtract(1).rename('CIRE'),
    image.normalizedDifference(['B6', 'B5']).rename('NDRE1'),
    image.normalizedDifference(['B7', 'B5']).rename('NDRE2'),
    b('B8').subtract(b('B4').multiply(1.35)).multiply(11.728).rename('LAI'),
    b('B8').divide(b('B3')).divide(b('B3').divide(b('B4'))).multiply(0.1134).add(0.14).rename('LCC'),
    b('B8').divide(b('B3')).subtract(1).multiply(0.8364).subtract(1.3231).rename('CCC')];
  var out = ee.Image.cat(idx);
  Object.keys(INDEX_CLAMP).forEach(function(k) {          // ratio indices explode where the denominator ~ 0
    var c = INDEX_CLAMP[k];
    out = out.addBands(out.select(k).max(c[0]).min(c[1]), null, true);
  });
  return image.addBands(out).copyProperties(image, ['system:time_start']);
}

function landsatLST(region) {                              // Landsat 8/9 surface temperature, deg C
  var col = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
    .filterBounds(region)
    .filterDate(ee.Date.fromYMD(START_YEAR, 1, 1), ee.Date.fromYMD(END_YEAR + 1, 1, 1))
    .filter(ee.Filter.calendarRange(SUMMER[0], SUMMER[1], 'month'));
  return col.map(function(img) {
    var qa = img.select('QA_PIXEL');
    var clear = qa.bitwiseAnd(parseInt('11111', 2)).eq(0);   // fill, dilated cloud, cirrus, cloud, shadow
    return img.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15).rename('Temperature')
      .updateMask(clear).copyProperties(img, ['system:time_start']);
  });
}

function seasonalComposite(col, year, months) {            // median of the months in one year, or all years
  var c = col.filter(ee.Filter.calendarRange(months[0], months[1], 'month'));
  if (year !== null) c = c.filter(ee.Filter.calendarRange(year, year, 'year'));
  return c.median();
}

function makeFeatures(region, cloudMax) {
  var s2 = s2Collection(region, cloudMax).map(addIndices);
  var summer = seasonalComposite(s2, null, SUMMER).clip(region);                  // 2019-2025 summer median
  var perYear = YEARS.map(function(y) { return seasonalComposite(s2, y, SUMMER).clip(region); });

  // rate of change of each index over the yearly summer medians START_YEAR..YEARS[k] (x = years since START_YEAR)
  function ratesTo(k) {
    if (k === 0) {                                           // one summer: no trend yet
      return ee.Image.constant(VI_NAMES.map(function() { return 0; })).float()
        .rename(VI_NAMES.map(function(v) { return v + '_Rate'; })).clip(region);
    }
    return ee.Image.cat(VI_NAMES.map(function(v) {
      var col = ee.ImageCollection.fromImages(YEARS.slice(0, k + 1).map(function(y, i) {
        return ee.Image.constant(y - START_YEAR).float().rename('t').addBands(perYear[i].select(v).rename('v'));
      }));
      var fit = RATE_METHOD === 'sens' ? col.reduce(ee.Reducer.sensSlope()).select('slope')
                                       : col.reduce(ee.Reducer.linearFit()).select('scale');
      return fit.rename(v + '_Rate');
    }));
  }
  var rates = ratesTo(YEARS.length - 1);

  // CSTV: standardised absolute change YEARS[k] - START_YEAR of the summer medians
  var first = perYear[0];
  var cstvOk = first.select('NDVI').gte(CSTV_NDVI_MIN);
  function cstvTo(k) {
    var last = perYear[k];
    return ee.Image.cat(VI_NAMES.map(function(v) {
      var d = last.select(v).subtract(first.select(v)).rename('d');
      var st = d.reduceRegion({reducer: ee.Reducer.mean().combine({reducer2: ee.Reducer.stdDev(), sharedInputs: true}),
                               geometry: region, scale: 20, maxPixels: 1e9, bestEffort: true});
      var mu = ee.Number(ee.Algorithms.If(st.get('d_mean'), st.get('d_mean'), 0));
      var sd = ee.Number(ee.Algorithms.If(st.get('d_stdDev'), st.get('d_stdDev'), 0)).max(1e-12);
      return d.subtract(mu).abs().divide(sd)
        .updateMask(cstvOk).unmask(CSTV_FILL).clip(region).rename('CSTV_' + v + '_Diff');
    }));
  }
  var cstv = cstvTo(YEARS.length - 1);
  if (ADD_CSTV_MASK) cstv = cstv.addBands(cstvOk.unmask(0).toFloat().clip(region).rename('CSTV_valid'));

  // static: terrain, soil, hydrology
  var dem = ee.Image('USGS/SRTMGL1_003').clip(region).rename('DEM');
  var terr = ee.Algorithms.Terrain(dem);
  var slope = terr.select('slope').rename('Slope_deg');
  var terrain = ee.Image.cat(dem, slope, terr.select('aspect').rename('Aspect_deg'),
    dem.convolve(ee.Kernel.fixed({width: 3, height: 3, weights: [[1, 1, 1], [1, -8, 1], [1, 1, 1]],
                                  normalize: false})).rename('Curvature'));
  var soil = ee.Image.cat(
    ee.Image('projects/soilgrids-isric/phh2o_mean').select('phh2o_0-5cm_mean').rename('pH'),
    ee.Image('projects/soilgrids-isric/sand_mean').select('sand_0-5cm_mean').rename('Sand_pct_0_5cm'),
    ee.Image('projects/soilgrids-isric/silt_mean').select('silt_0-5cm_mean').rename('Silt_pct_0_5cm'),
    ee.Image('projects/soilgrids-isric/clay_mean').select('clay_0-5cm_mean').rename('Clay_pct_0_5cm')).clip(region);
  var merit = ee.Image('MERIT/Hydro/v1_0_1').clip(region);
  var upa = merit.select('upa').rename('UpstreamArea_km2');
  var twi = upa.multiply(1e6).divide(ee.Image.pixelArea().sqrt())
    .divide(slope.multiply(Math.PI / 180).tan().add(1e-6)).max(1e-6).log().rename('TWI');
  var hydro = ee.Image.cat(upa, merit.select('hnd').rename('HAND_m'), twi);
  var staticBands = ee.Image.cat(terrain, soil, hydro);

  // climate
  var lst = landsatLST(region);
  var lstAll = lst.median().clip(region);
  function coarseClimate(y) {                               // y = null: all years, all seasons (previous bands)
    var d0 = ee.Date.fromYMD(y === null ? START_YEAR : y, y === null ? 1 : SUMMER[0], 1);
    var d1 = y === null ? ee.Date.fromYMD(END_YEAR + 1, 1, 1) : ee.Date.fromYMD(y, SUMMER[1] + 1, 1);
    var temp = ee.ImageCollection('MODIS/061/MOD11A1').filterDate(d0, d1).select('LST_Day_1km').mean()
      .multiply(0.02).subtract(273.15).rename('Temperature');
    var hum = ee.ImageCollection('NASA/NLDAS/FORA0125_H002').filterDate(d0, d1).select('specific_humidity')
      .mean().rename('Humidity');
    var moist = ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H').filterDate(d0, d1)
      .select('SoilMoi0_10cm_inst').mean().rename('Moisture');
    var years_ = y === null ? YEARS : [y];
    var precip = ee.ImageCollection(years_.map(function(yr) {
      return ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
        .filterDate(ee.Date.fromYMD(yr, SUMMER[0], 1), ee.Date.fromYMD(yr, SUMMER[1] + 1, 1))
        .select('precipitation').sum();
    })).mean().rename('PrecipJJA_mm_mean');
    return {temp: temp, hum: hum, moist: moist, precip: precip};
  }
  var climate = null;
  if (STATIC_CLIMATE === 'landsat') climate = lstAll;
  if (STATIC_CLIMATE === 'coarse') {
    var cc = coarseClimate(null);
    climate = ee.Image.cat(cc.temp, cc.hum, cc.moist, cc.precip).clip(region);
  }

  // Dynamic World (summer): class probabilities, modal class, vegetation probability
  function dw(y) {
    var c = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filterBounds(region)
      .filterDate(ee.Date.fromYMD(START_YEAR, 1, 1), ee.Date.fromYMD(END_YEAR + 1, 1, 1))
      .filter(ee.Filter.calendarRange(SUMMER[0], SUMMER[1], 'month'));
    if (y !== null) c = c.filter(ee.Filter.calendarRange(y, y, 'year'));
    var prob = c.select(DW_SRC).mean().rename(DW_NAMES);
    return ee.Image.cat(prob, c.select('label').mode().rename('DW_LandCover'),
      prob.select(['DW_Trees', 'DW_Grass', 'DW_FloodedVeg', 'DW_Crops', 'DW_Shrubs'])
        .reduce(ee.Reducer.sum()).rename('DW_Vegetation')).clip(region);
  }

  // seasonal features (optional)
  var seasonal = null;
  if (SEASONAL_FEATURES === 'seasonal' || SEASONAL_FEATURES === 'both') {
    var spring = seasonalComposite(s2, null, SPRING).select('NDVI');
    var autumn = seasonalComposite(s2, null, AUTUMN).select('NDVI');
    seasonal = ee.Image.cat(spring.rename('NDVI_spring'), autumn.rename('NDVI_autumn'),
      spring.subtract(summer.select('NDVI')).rename('NDVI_spring_minus_summer'),
      autumn.subtract(summer.select('NDVI')).rename('NDVI_autumn_minus_summer')).clip(region);
  }
  if (SEASONAL_FEATURES === 'harmonic' || SEASONAL_FEATURES === 'both') {
    var harm = s2.select('NDVI').map(function(img) {
      var t = ee.Date(img.get('system:time_start')).difference(ee.Date.fromYMD(START_YEAR, 1, 1), 'year');
      var w = t.multiply(2 * Math.PI);
      return ee.Image.cat(ee.Image.constant(1), ee.Image.constant(t), ee.Image.constant(w.cos()),
                          ee.Image.constant(w.sin()), img).float();
    }).reduce(ee.Reducer.linearRegression(4, 1)).select('coefficients').arrayProject([0])
      .arrayFlatten([['c0', 'c_t', 'c_cos', 'c_sin']]);
    var hb = ee.Image.cat(harm.select('c_cos').hypot(harm.select('c_sin')).rename('NDVI_harm_amplitude'),
      harm.select('c_sin').atan2(harm.select('c_cos')).rename('NDVI_harm_phase')).clip(region);
    seasonal = seasonal === null ? hb : seasonal.addBands(hb);
  }

  // main stack: multi-year summer composite + change, static, climate, rates, land cover
  var stack = summer.select(VI_NAMES).addBands(summer.select(S2_BANDS)).addBands(summer.select('QA60'))
    .addBands(cstv).addBands(terrain);
  if (climate !== null) stack = stack.addBands(climate);
  stack = stack.addBands(soil).addBands(hydro).addBands(rates).addBands(dw(null));
  if (seasonal !== null) stack = stack.addBands(seasonal);
  if (STACK_LAYOUT === 'hazardfuse66') stack = stack.select(BANDS_66);

  // yearly stacks. 'hazardfuse66': the same 66 bands as the main stack - that summer's S2 bands, indices, QA60,
  // Dynamic World and climate (Temperature, Humidity, Moisture, PrecipJJA_mm_mean of that summer), CSTV and rates
  // per YEARLY_CHANGE, and the static terrain, soil and hydrology bands. 'extended': the previous yearly layout.
  var yearly = YEARS.map(function(y, i) {
    if (STACK_LAYOUT === 'hazardfuse66') {
      var cy = coarseClimate(y);
      var k = YEARLY_CHANGE === 'to_date' ? i : YEARS.length - 1;
      var last = k === YEARS.length - 1;                     // last year (or 'full'): the main-stack values
      var img66 = perYear[i].select(VI_NAMES).addBands(perYear[i].select(S2_BANDS)).addBands(perYear[i].select('QA60'))
        .addBands(last ? cstv : cstvTo(k))
        .addBands(terrain).addBands(cy.temp).addBands(cy.hum).addBands(cy.moist).addBands(soil)
        .addBands(cy.precip).addBands(hydro).addBands(last ? rates : ratesTo(k))
        .addBands(dw(y));
      return img66.select(BANDS_66).clip(region).toFloat().set('year', y);
    }
    var img = perYear[i].select(VI_NAMES).addBands(perYear[i].select(S2_BANDS)).addBands(dw(y));
    if (YEARLY_FORCING) {
      var cy2 = coarseClimate(y);
      var lc = lst.filter(ee.Filter.calendarRange(y, y, 'year'));                         // gaps: all-years LST
      var lstY = ee.Image(ee.Algorithms.If(lc.size().gt(0), lc.median().unmask(lstAll), lstAll));
      img = img.addBands(STATIC_CLIMATE === 'coarse' ? cy2.temp : lstY.rename('Temperature'))
        .addBands(cy2.hum).addBands(cy2.precip);
    }
    return img.addBands(staticBands).clip(region).toFloat().set('year', y);
  });
  return {featureImage: stack.toFloat(), yearly: yearly, summer: summer, perYear: perYear, rates: rates};
}

// =====================================================================
// BUILD, PRINT FEATURE NAMES, EXPORT
// =====================================================================
var F = makeFeatures(roi);
var featureImage = F.featureImage;
var siteKey = selectedSite.replace(/ /g, '_').toLowerCase();

// numbered list of band names, printed as one text block (readable without expanding the list)
function printNames(img, title) {
  img.bandNames().evaluate(function(names, err) {
    if (err || !names) { print(title + ': band names not available:', err); return; }
    var lines = names.map(function(n, i) { return (i + 1 < 10 ? ' ' : '') + (i + 1) + '. ' + n; });
    print(title + ' (' + names.length + ' bands):\n' + lines.join('\n'));
  });
}
printNames(featureImage, 'Main stack ' + siteKey + '_Features');
printNames(F.yearly[0], 'Yearly stacks ' + siteKey + '_FeatureStack_<year> (same bands every year)');
if (STACK_LAYOUT === 'hazardfuse66') {                       // every stack must match the 66-band list exactly
  ee.List([featureImage].concat(F.yearly).map(function(img) {
    return img.bandNames().equals(ee.List(BANDS_66));
  })).evaluate(function(ok, err) {
    if (err) { print('Band check not computed:', err); return; }
    var all = ok.every(function(v) { return v; });
    print('66-band layout (names and order of the original stack):',
          all ? 'OK - main + ' + F.yearly.length + ' yearly stacks' : 'MISMATCH ' + JSON.stringify(ok));
  });
}
print('Settings: years', START_YEAR + '-' + END_YEAR, '| summer months', SUMMER.join('-'), '| scene filter <',
      SCENE_CLOUD_MAX + '% | cloud mask', CLOUD_MASK, CLOUD_MASK === 'csplus' ? '(cs_cdf >= ' + CS_THRESHOLD + ')' : '',
      '| climate', STATIC_CLIMATE, '| layout', STACK_LAYOUT,
      STACK_LAYOUT === 'hazardfuse66' ? '| yearly CSTV/rates: ' + YEARLY_CHANGE : '| seasonal ' + SEASONAL_FEATURES);

// =====================================================================
// MAP DISPLAY SETTINGS: EDIT here
// ---------------------------------------------------------------------
// Every band of the chosen stack is added to the map, clipped to the ROI,
// in stack order ("1. NDVI" ... "66. DW_Vegetation"); tick a layer in the
// Layers list to show it. Bands with known limits use fixed ranges
// (VIS_FIXED); all other bands are stretched between the VIS_PERCENTILES
// of their values inside the ROI. The stretch is computed on the server,
// so the layers appear at once. Click a pixel with the Inspector to read
// the band values ("All bands (values)" layer).
// =====================================================================
var VIS_STACK = 'main';               // 'main' (<site>_Features) or a year, e.g. 2024 (<site>_FeatureStack_2024)
var VIS_SHOW = ['NDVI'];              // band layers switched on when the script runs
var VIS_SHOW_TRUE_COLOUR = true;      // true-colour composite switched on
var VIS_PERCENTILES = [2, 98];        // stretch of the bands without a fixed range
var VIS_STATS_SCALE = 30;             // m; scale of the stretch statistics
var VIS_OPACITY = 1.0;
var VIS_ROI_COLOR = 'FFFFFF', VIS_ROI_WIDTH = 2;   // ROI outline, drawn on top
var VIS_PALETTES = {                  // colours per band group (hex without '#'), low -> high
  index:       ['8c510a', 'd8b365', 'f6e8c3', 'c7eae5', '5ab4ac', '01665e'],   // vegetation indices
  reflectance: ['000000', '404040', '808080', 'bfbfbf', 'ffffff'],             // S2 bands B1-B12
  qa:          ['ffffff', 'fdae61', 'd7191c'],                                 // QA60: 0 clear, 1024 opaque, 2048 cirrus
  cstv:        ['ffffff', 'fee8c8', 'fdbb84', 'e34a33', 'b30000'],             // CSTV standardised change
  rate:        ['b2182b', 'ef8a62', 'f7f7f7', '67a9cf', '2166ac'],             // Sen's slope; decline red, gain blue
  elevation:   ['1a9850', '91cf60', 'd9ef8b', 'fee08b', 'fc8d59', 'd73027'],
  slope:       ['ffffcc', 'fed976', 'fd8d3c', 'e31a1c', '800026'],
  aspect:      ['ff0000', 'ffff00', '00ff00', '00ffff', '0000ff', 'ff00ff', 'ff0000'],   // circular: N-E-S-W-N
  diverging:   ['2166ac', '67a9cf', 'f7f7f7', 'ef8a62', 'b2182b'],             // curvature
  temperature: ['313695', '74add1', 'ffffbf', 'f46d43', 'a50026'],
  moisture:    ['f7fbff', 'c6dbef', '6baed6', '2171b5', '08306b'],             // humidity, soil moisture, precipitation
  soil:        ['fff7ec', 'fdd49e', 'fc8d59', 'd7301f', '7f0000'],             // pH, sand, silt, clay
  hydro:       ['ffffd9', 'c7e9b4', '41b6c4', '225ea8', '081d58'],             // upstream area, HAND, TWI
  dw_prob:     ['ffffff', '6a51a3'],                                           // Dynamic World class probabilities
  vegetation:  ['ffffff', '74c476', '00441b'],                                 // DW_Vegetation
  dw_class:    ['419bdf', '397d49', '88b053', '7a87c6', 'e49635', 'dfc35a', 'c4281b', 'a59b8f', 'b39fe1'],
  label:       ['008000', 'd73027'],                                           // class 0, class 1 (EXPORT_LABELS)
  hi:          ['1a9850', 'fee08b', 'd73027'],                                 // kriged HI (EXPORT_LABELS)
  other:       ['440154', '3b528b', '21918c', '5ec962', 'fde725']
};
var VIS_FIXED = {                     // fixed display ranges [min, max] (all other bands: percentile stretch)
  QA60: [0, 2048], Aspect_deg: [0, 360], DW_LandCover: [0, 8], DW_Vegetation: [0, 1],
  DW_Water: [0, 1], DW_Trees: [0, 1], DW_Grass: [0, 1], DW_FloodedVeg: [0, 1], DW_Crops: [0, 1], DW_Shrubs: [0, 1],
  DW_Built: [0, 1], DW_Bare: [0, 1], DW_Snow: [0, 1]
};
var VIS_SYMMETRIC = ['rate', 'diverging'];   // groups stretched symmetrically about 0
var VIS_TRUE_COLOUR = {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.3, gamma: 1.2};              // reflectance 0-1
var VIS_FALSE_COLOUR = {bands: ['B8', 'B4', 'B3'], min: 0, max: [0.45, 0.3, 0.3], gamma: 1.2};

// =====================================================================
// MAP DISPLAY
// =====================================================================
function visGroup(b) {
  if (b === 'DW_LandCover') return 'dw_class';
  if (b === 'DW_Vegetation') return 'vegetation';
  if (b === 'QA60') return 'qa';
  if (/_Rate$/.test(b)) return 'rate';
  if (/^CSTV_/.test(b)) return 'cstv';
  if (/^DW_/.test(b)) return 'dw_prob';
  if (VI_NAMES.indexOf(b) >= 0) return 'index';
  if (S2_BANDS.indexOf(b) >= 0) return 'reflectance';
  return {DEM: 'elevation', Slope_deg: 'slope', Aspect_deg: 'aspect', Curvature: 'diverging',
          Temperature: 'temperature', Humidity: 'moisture', Moisture: 'moisture', PrecipJJA_mm_mean: 'moisture',
          pH: 'soil', Sand_pct_0_5cm: 'soil', Silt_pct_0_5cm: 'soil', Clay_pct_0_5cm: 'soil',
          UpstreamArea_km2: 'hydro', HAND_m: 'hydro', TWI: 'hydro'}[b] || 'other';
}

// band b as a 0-1 image (fixed range or ROI percentile stretch), for display with {min: 0, max: 1}
function displayBand(img, b) {
  var band = img.select(b);
  if (VIS_FIXED[b]) return band.unitScale(VIS_FIXED[b][0], VIS_FIXED[b][1]).clamp(0, 1);
  var st = band.reduceRegion({reducer: ee.Reducer.percentile(VIS_PERCENTILES), geometry: roi,
                              scale: VIS_STATS_SCALE, maxPixels: 1e10, bestEffort: true, tileScale: 4});
  var get = function(p) {                                     // percentile, 0 if the band has no valid pixel
    var v = st.get(b + '_p' + p);
    return ee.Number(ee.Algorithms.If(ee.Algorithms.IsEqual(v, null), 0, v));
  };
  var lo = get(VIS_PERCENTILES[0]), hi = get(VIS_PERCENTILES[1]);
  if (VIS_SYMMETRIC.indexOf(visGroup(b)) >= 0) {
    var m = lo.abs().max(hi.abs()).max(1e-9);
    lo = m.multiply(-1); hi = m;
  }
  hi = hi.max(lo.add(1e-9));                                  // constant band inside the ROI
  return band.unitScale(lo, hi).clamp(0, 1);
}

var VIS_YEAR_INDEX = YEARS.indexOf(VIS_STACK);
if (VIS_STACK !== 'main' && VIS_YEAR_INDEX < 0) {
  print('VIS_STACK = ' + VIS_STACK + ' is not one of ' + YEARS.join(', ') + ': map display skipped.');
} else {
  var visImg = (VIS_STACK === 'main' ? featureImage : F.yearly[VIS_YEAR_INDEX]).clip(roi);
  var visPrefix = VIS_STACK === 'main' ? '' : VIS_STACK + ' ';
  Map.addLayer(visImg, VIS_TRUE_COLOUR, visPrefix + 'True colour (B4, B3, B2)', VIS_SHOW_TRUE_COLOUR);
  Map.addLayer(visImg, VIS_FALSE_COLOUR, visPrefix + 'False colour (B8, B4, B3)', false);
  var visNames = STACK_LAYOUT === 'hazardfuse66' ? BANDS_66 : null;
  var addBands = function(names) {
    names.forEach(function(b, i) {
      var g = visGroup(b);
      var vis = {min: 0, max: 1, palette: VIS_PALETTES[g]};
      if (g === 'dw_class') {                                  // categorical: one colour per class, no stretch
        Map.addLayer(visImg.select(b), {min: 0, max: 8, palette: VIS_PALETTES.dw_class},
                     visPrefix + (i + 1) + '. ' + b, VIS_SHOW.indexOf(b) >= 0, VIS_OPACITY);
        return;
      }
      Map.addLayer(displayBand(visImg, b), vis, visPrefix + (i + 1) + '. ' + b, VIS_SHOW.indexOf(b) >= 0, VIS_OPACITY);
    });
    Map.addLayer(visImg, {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.3}, visPrefix + 'All bands (values)', true, 0);   // transparent; Inspector
    Map.addLayer(ee.Image().paint(roi, 1, VIS_ROI_WIDTH), {palette: [VIS_ROI_COLOR]}, 'ROI outline');
    print('Map: ' + names.length + ' bands of ' + (VIS_STACK === 'main' ? siteKey + '_Features'
          : siteKey + '_FeatureStack_' + VIS_STACK) + ' added (Layers list). Fixed ranges: ' + Object.keys(VIS_FIXED).join(', ')
          + '; other bands: ' + VIS_PERCENTILES.join('-') + ' percentile stretch inside the ROI. Pixel values: Inspector.');
  };
  if (visNames) addBands(visNames);
  else visImg.bandNames().evaluate(function(n) { addBands(n); });   // 'extended' layout: names from the server
}

function exportImg(img, name) {
  Export.image.toDrive({image: img, description: name, folder: EXPORT_FOLDER, fileNamePrefix: name,
    region: roi, crs: EXPORT_CRS, scale: EXPORT_SCALE, maxPixels: 1e13, fileFormat: 'GeoTIFF'});
}
var tasks = [];
if (EXPORT_MAIN) { exportImg(featureImage, siteKey + '_Features'); tasks.push(siteKey + '_Features'); }
if (EXPORT_YEARLY) {
  YEARS.forEach(function(y, i) {
    exportImg(F.yearly[i], siteKey + '_FeatureStack_' + y);
    tasks.push(siteKey + '_FeatureStack_' + y);
  });
}

// optional: labels and continuous HI on the same grid (needs the well asset and kriging)
if (EXPORT_LABELS) {
  var krige = function(asset) {
    return ee.FeatureCollection(asset).kriging({
      propertyName: 'HI', shape: 'exponential', range: cfg.kriging.range, sill: cfg.kriging.sill,
      nugget: cfg.kriging.nugget, maxDistance: cfg.kriging.maxDistance, reducer: ee.Reducer.mean()
    }).rename('HI_kriged').clip(roi);
  };
  var krigedHI = krige(cfg.asset);
  var label = makeLabel(krigedHI, roi, cfg.splitHI).label;
  Map.addLayer(krigedHI, {min: 0, max: cfg.rawMax, palette: VIS_PALETTES.hi}, 'HI kriged (0-' + cfg.rawMax + ')', false);
  Map.addLayer(label, {min: 0, max: 1, palette: VIS_PALETTES.label}, 'Label (0 lower HI, 1 higher HI)', false);
  exportImg(label.toByte(), siteKey + '_Label'); tasks.push(siteKey + '_Label');
  exportImg(krigedHI.toFloat(), siteKey + '_HI'); tasks.push(siteKey + '_HI');
  if (EXPORT_YEARLY) {
    YEARS.forEach(function(y) {
      var labY = YEARLY_LABEL_ASSETS[y] ? makeLabel(krige(YEARLY_LABEL_ASSETS[y]), roi, cfg.splitHI).label : label;
      exportImg(labY.toByte(), siteKey + '_Label_' + y); tasks.push(siteKey + '_Label_' + y);
    });
  }
}
print('Export tasks (' + tasks.length + ', run them in the Tasks tab) to Drive/' + EXPORT_FOLDER + ':', tasks.join(', '));