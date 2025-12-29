# ================= 配置部分 =================

# 导入必要的库
from openai import OpenAI

# Gemini模型API配置
GEMINI_API_KEY = "sk-hY1PLRISvYRksP0HNJELF2NIv3oqTeW07wAEO0ak432VHHDf"

# DeepSeek模型API配置
DEEPSEEK_API_KEY = "sk-nuzywtfwqsmxwwgheoftmtajhdqmrryqqcisciaxkggzqibz"

# ⚡️⚡️⚡️ 关键修改：创建两个独立的客户端实例
# Gemini客户端
client_gemini = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://www.dmxapi.cn/v1",
    timeout=60.0,
    max_retries=0
)

# DeepSeek客户端
client_deepseek = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.siliconflow.cn/v1",
    timeout=60.0,
    max_retries=0
)

# 🤖 模型配置
# DIRECTOR_MODEL = "inclusionAI/Ling-flash-2.0" 
DIRECTOR_MODEL = "gemini-3-flash-preview" 
# HELPER_MODEL = "Qwen/Qwen3-235B-A22B" 
HELPER_MODEL = "deepseek-ai/DeepSeek-V3.2" 
# HELPER_MODEL = "moonshotai/Kimi-K2-Thinking" 
VERSION = "v80_unique"

# 测试模式开关："test" 表示测试模式，随机选择5个词生成情景
TEST_MODE = "all" 

SCENARIO_TYPES = [
    "CG Scenario"          # 后期CG：超现实、魔法、违反物理
]

# ================= MacGuffin Library (Expanded to ~400 items) =================

# 包含约 400 个常见物体，分为 12 个大类，涵盖不同材质、形态、大小和物理属性
OBJECT_DICT = [
    # -----------------------------------------------------------
    # 1. 🍎 食品与饮料 (Food & Drinks) - [~60 items]
    # -----------------------------------------------------------
    "Apple (苹果)", "Banana (香蕉)", "Orange (橙子)", "Grape (葡萄)", "Strawberry (草莓)",
    "Watermelon (西瓜)", "Pineapple (菠萝)", "Kiwi (猕猴桃)", "Mango (芒果)", "Avocado (牛油果)",
    "Tomato (番茄)", "Potato (土豆)", "Carrot (胡萝卜)", "Onion (洋葱)", "Garlic (大蒜)",
    "Broccoli (西兰花)", "Cucumber (黄瓜)", "Corn (玉米)", "Red Pepper (红椒)", "Egg (鸡蛋)",
    "Fried Egg (煎蛋)", "Boiled Egg (水煮蛋)", "Bread Loaf (面包)", "Croissant (羊角面包)", "Bagel (贝果)",
    "Pancake (薄煎饼)", "Waffle (华夫饼)", "Pizza Slice (披萨切片)", "Burger (汉堡)", "Hot Dog (热狗)",
    "Sandwich (三明治)", "Sushi Roll (寿司)", "Taco (塔可)", "Noodle Bowl (面条)", "Steak (牛排)",
    "Chicken Leg (鸡腿)", "Shrimp (虾)", "Cheese Block (奶酪)", "Butter Stick (黄油)", "Yogurt Cup (酸奶)",
    "Ice Cream Cone (冰淇淋甜筒)", "Birthday Cake (生日蛋糕)", "Cookie (曲奇饼干)", "Chocolate Bar (巧克力棒)", "Donut (甜甜圈)",
    "Popcorn Bucket (爆米花)", "Potato Chips (薯片)", "Lollipop (棒棒糖)", "Marshmallow (棉花糖)", "Jelly/Jello (果冻)",
    "Soda Can (易拉罐汽水)", "Juice Box (盒装果汁)", "Coffee Cup (咖啡杯)", "Milk Carton (牛奶盒)", "Wine Bottle (酒瓶)",

    # -----------------------------------------------------------
    # 2. 🏠 家居与日用 (Household & Daily Items) - [~50 items]
    # -----------------------------------------------------------
    "Wooden Chair (木椅)", "Office Chair (办公椅)", "Table (桌子)", "Sofa (沙发)", "Bed (床)",
    "Pillow (枕头)", "Blanket (毯子)", "Floor Lamp (落地灯)", "Desk Lamp (台灯)", "Rug (地毯)",
    "Curtain (窗帘)", "Mirror (镜子)", "Wall Clock (挂钟)", "Alarm Clock (闹钟)", "Vase (花瓶)",
    "Picture Frame (相框)", "Candle (蜡烛)", "Book (书)", "Magazine (杂志)", "Newspaper (报纸)",
    "Remote Control (遥控器)", "Key (钥匙)", "Wallet (钱包)", "Coin (硬币)", "Umbrella (雨伞)",
    "Backpack (背包)", "Suitcase (行李箱)", "Trash Can (垃圾桶)", "Broom (扫帚)", "Mop (拖把)",
    "Bucket (水桶)", "Sponge (海绵)", "Towel (毛巾)", "Soap Bar (肥皂)", "Shampoo Bottle (洗发水瓶)",
    "Toothbrush (牙刷)", "Toothpaste (牙膏)", "Toilet Paper Roll (卷纸)", "Tissue Box (纸巾盒)", "Laundry Basket (洗衣篮)",
    "Coat Hanger (衣架)", "Iron (熨斗)", "Vacuum Cleaner (吸尘器)", "Electric Fan (电风扇)", "Heater (取暖器)",
    "Light Bulb (灯泡)", "Power Strip (插线板)", "Battery (电池)", "Matchbox (火柴盒)", "Lighter (打火机)",

    # -----------------------------------------------------------
    # 3. 🍴 厨房用具 (Kitchenware) - [~35 items]
    # -----------------------------------------------------------
    "Ceramic Plate (陶瓷盘)", "Bowl (碗)", "Glass Cup (玻璃杯)", "Mug (马克杯)", "Wine Glass (高脚杯)",
    "Fork (叉子)", "Spoon (勺子)", "Knife (餐刀)", "Chopsticks (筷子)", "Frying Pan (煎锅)",
    "Cooking Pot (汤锅)", "Kettle (水壶)", "Teapot (茶壶)", "Cutting Board (砧板)", "Grater (擦丝器)",
    "Peeler (削皮器)", "Whisk (打蛋器)", "Spatula (锅铲)", "Ladle (汤勺)", "Kitchen Tongs (食品夹)",
    "Can Opener (开罐器)", "Corkscrew (开瓶器)", "Measuring Cup (量杯)", "Kitchen Scale (电子秤)", "Timer (计时器)",
    "Stand Mixer (厨师机)", "Blender (搅拌机)", "Toaster (烤面包机)", "Microwave (微波炉)", "Oven (烤箱)",
    "Refrigerator (冰箱)", "Dishwasher (洗碗机)", "Thermos (保温杯)", "Lunchbox (饭盒)", "Salt Shaker (盐瓶)",

    # -----------------------------------------------------------
    # 4. 💻 电子与科技 (Electronics & Tech) - [~30 items]
    # -----------------------------------------------------------
    "Smartphone (智能手机)", "Tablet (平板电脑)", "Laptop (笔记本电脑)", "Desktop Monitor (显示器)", "Mechanical Keyboard (机械键盘)",
    "Computer Mouse (鼠标)", "Printer (打印机)", "Router (路由器)", "Webcam (摄像头)", "Microphone (麦克风)",
    "Headphones (头戴式耳机)", "Earbuds (入耳式耳机)", "Bluetooth Speaker (蓝牙音箱)", "DSLR Camera (单反相机)", "Camera Lens (镜头)",
    "Tripod (三脚架)", "USB Flash Drive (U盘)", "SD Card (存储卡)", "Game Console (游戏主机)", "Game Controller (手柄)",
    "VR Headset (VR头显)", "Smartwatch (智能手表)", "Fitness Tracker (手环)", "Calculator (计算器)", "Television (电视机)",
    "Projector (投影仪)", "Drone (无人机)", "Robot Vacuum (扫地机器人)", "Retro Gameboy (复古游戏机)", "Walkie Talkie (对讲机)",

    # -----------------------------------------------------------
    # 5. ✏️ 办公与文具 (Office & Stationery) - [~30 items]
    # -----------------------------------------------------------
    "Ballpoint Pen (圆珠笔)", "Fountain Pen (钢笔)", "Pencil (铅笔)", "Eraser (橡皮)", "Pencil Sharpener (卷笔刀)",
    "Ruler (尺子)", "Scissors (剪刀)", "Glue Stick (固体胶)", "Scotch Tape (透明胶带)", "Stapler (订书机)",
    "Paper Clip (回形针)", "Binder Clip (长尾夹)", "Thumbtack (图钉)", "Rubber Band (橡皮筋)", "Notebook (笔记本)",
    "Diary (日记本)", "File Folder (文件夹)", "Envelope (信封)", "Post-it Note (便利贴)", "Whiteboard Marker (白板笔)",
    "Chalk (粉笔)", "Blackboard (黑板)", "Globe (地球仪)", "Map (地图)", "Magnifying Glass (放大镜)",
    "Microscope (显微镜)", "Telescope (望远镜)", "Compass (指南针)", "Protractor (量角器)", "Clipboard (写字板)",

    # -----------------------------------------------------------
    # 6. 👗 服饰与配饰 (Clothing & Accessories) - [~40 items]
    # -----------------------------------------------------------
    "T-shirt (T恤)", "Dress Shirt (衬衫)", "Sweater (毛衣)", "Hoodie (卫衣)", "Jacket (夹克)",
    "Coat (大衣)", "Vest (背心)", "Jeans (牛仔裤)", "Trousers (西裤)", "Shorts (短裤)",
    "Skirt (短裙)", "Dress (连衣裙)", "Suit (西装)", "Necktie (领带)", "Bowtie (领结)",
    "Scarf (围巾)", "Gloves (手套)", "Mittens (连指手套)", "Baseball Cap (棒球帽)", "Beanie (毛线帽)",
    "Fedora (礼帽)", "Socks (袜子)", "Sneakers (运动鞋)", "High Heels (高跟鞋)", "Boots (靴子)",
    "Sandals (凉鞋)", "Slippers (拖鞋)", "Leather Belt (皮带)", "Wristwatch (手表)", "Diamond Ring (钻戒)",
    "Gold Necklace (金项链)", "Earrings (耳环)", "Glasses (眼镜)", "Sunglasses (太阳镜)",
    "Handbag (手提包)", "Tote Bag (托特包)", "Purse (零钱包)", "Hair Clip (发夹)", "Perfume Bottle (香水瓶)",

    # -----------------------------------------------------------
    # 7. 🧸 玩具与游戏 (Toys & Games) - [~30 items]
    # -----------------------------------------------------------
    "Porcelain Doll (瓷娃娃)", "Teddy Bear (泰迪熊)", "Action Figure (手办)", "Robot Toy (机器人玩具)", "Toy Car (玩具车)",
    "Toy Train (玩具火车)", "Paper Airplane (纸飞机)", "Rubber Duck (橡皮鸭)", "Lego Brick (乐高积木)", "Jigsaw Puzzle (拼图)",
    "Chess Board (棋盘)", "Playing Cards (扑克牌)", "Dice (骰子)", "Glass Marble (玻璃弹珠)", "Yo-yo (溜溜球)",
    "Spinning Top (陀螺)", "Kite (风筝)", "Balloon (气球)", "Soap Bubbles (肥皂泡)", "Slime (史莱姆)",
    "Play-Doh (橡皮泥)", "Water Gun (水枪)", "Nerf Gun (软弹枪)", "Jump Rope (跳绳)", "Hula Hoop (呼啦圈)",
    "Skateboard (滑板)", "Roller Skates (轮滑鞋)", "Bicycle (自行车)", "Scooter (滑板车)", "Slinky (弹簧玩具)",

    # -----------------------------------------------------------
    # 8. ⚽ 体育器材 (Sports Equipment) - [~20 items]
    # -----------------------------------------------------------
    "Soccer Ball (足球)", "Basketball (篮球)", "American Football (橄榄球)", "Baseball (棒球)", "Tennis Ball (网球)",
    "Golf Ball (高尔夫球)", "Ping Pong Ball (乒乓球)", "Volleyball (排球)", "Bowling Ball (保龄球)", "Badminton Shuttlecock (羽毛球)",
    "Tennis Racket (网球拍)", "Baseball Bat (棒球棍)", "Golf Club (高尔夫球杆)", "Hockey Stick (曲棍球杆)", "Helmet (头盔)",
    "Sports Jersey (球衣)", "Whistle (哨子)", "Trophy (奖杯)", "Gold Medal (金牌)", "Dumbbell (哑铃)",

    # -----------------------------------------------------------
    # 9. 🎸 乐器 (Musical Instruments) - [~20 items]
    # -----------------------------------------------------------
    "Acoustic Guitar (吉他)", "Violin (小提琴)", "Cello (大提琴)", "Grand Piano (钢琴)", "Electronic Keyboard (电子琴)",
    "Drum Kit (架子鼓)", "Flute (长笛)", "Clarinet (单簧管)", "Saxophone (萨克斯)", "Trumpet (小号)",
    "Trombone (长号)", "Tuba (大号)", "Harmonica (口琴)", "Accordion (手风琴)", "Xylophone (木琴)",
    "Tambourine (铃鼓)", "Maracas (沙锤)", "Triangle (三角铁)", "Cymbal (镲)", "Metronome (节拍器)",

    # -----------------------------------------------------------
    # 10. 🔧 工具与五金 (Tools & Hardware) - [~30 items]
    # -----------------------------------------------------------
    "Hammer (锤子)", "Screwdriver (螺丝刀)", "Wrench (扳手)", "Pliers (钳子)", "Hand Saw (手锯)",
    "Electric Drill (电钻)", "Tape Measure (卷尺)", "Spirit Level (水平仪)", "Iron Nail (钉子)", "Screw (螺丝)",
    "Bolt and Nut (螺栓螺母)", "Metal Washer (垫圈)", "Door Hinge (合页)", "Metal Hook (挂钩)", "Padlock (挂锁)",
    "Metal Chain (铁链)", "Rope (绳子)", "Ladder (梯子)", "Shovel (铁锹)", "Rake (耙子)",
    "Garden Hoe (锄头)", "Axe (斧头)", "Wheelbarrow (独轮车)", "Watering Can (洒水壶)", "Garden Hose (水管)",
    "Lawn Mower (割草机)", "Paintbrush (油漆刷)", "Paint Roller (滚筒刷)", "Flashlight (手电筒)", "Duct Tape (胶带)",

    # -----------------------------------------------------------
    # 11. 🌲 自然与材质 (Nature & Materials) - [~30 items]
    # -----------------------------------------------------------
    "Red Rose (红玫瑰)", "Sunflower (向日葵)", "Flower Bouquet (花束)",
    "Cactus (仙人掌)", "Green Leaf (绿叶)", "Tree Branch (树枝)", "Wooden Log (原木)", "Tree Stump (树桩)",
    "Rock (岩石)", "Smooth Stone (鹅卵石)", "Crystal Geode (水晶洞)", "Seashell (贝壳)", "Pinecone (松果)",
    "Acorn (橡果)", "Mushroom (蘑菇)", "Feather (羽毛)", "Bird Nest (鸟巢)", "Spider Web (蜘蛛网)",
    "Sand Pile (沙堆)", "Soil (泥土)", "Mud Puddle (泥坑)", "Water Drop (水滴)", "Snowflake (雪花)",
    "Icicle (冰柱)", "Cloud (云朵)", "Sun (太阳)", "Moon (月亮)", "Star (星星)",

    # -----------------------------------------------------------
    # 12. 📦 杂项与包装 (Miscellaneous) - [~30 items]
    # -----------------------------------------------------------
    "Cardboard Box (纸箱)", "Milk Crate (牛奶箱)", "Plastic Bag (塑料袋)", "Burlap Sack (麻袋)", "Glass Bottle (玻璃瓶)",
    "Mason Jar (梅森罐)", "Tin Can (锡罐)", "Toothpaste Tube (牙膏管)", "Gift Box (礼品盒)", "Stack of Books (书堆)",
    "Pile of Clothes (衣服堆)", "Road Barrier (路障)", "Stop Sign (停止标志)", "Flag (旗帜)",
    "Banner (横幅)", "Poster (海报)", "Sticker (贴纸)", "Badge (徽章)", "Ticket (票)",
    "Credit Card (信用卡)", "Passport (护照)", "ID Card (身份证)", "Stack of Money (钞票堆)", "Gold Bar (金条)",
    "Diamond (钻石)", "Brick (砖头)", "Tire (轮胎)", "Mannequin (人体模型)",

    # -----------------------------------------------------------
    # 13. 🚗 交通工具 (Vehicles) - [~30 items]
    # -----------------------------------------------------------
    "Bicycle (自行车)", "Motorcycle (摩托车)", "Car (汽车)", "Bus (公交车)", "Truck (卡车)",
    "Train (火车)", "Airplane (飞机)", "Helicopter (直升机)", "Ship (轮船)", "Submarine (潜艇)",
    "Motorcycle Helmet (摩托车头盔)", "Car Wheel (汽车轮胎)", "Traffic Light (交通灯)", "Fuel Pump (加油站)",
    "Parking Meter (停车计时器)", "Car Door (汽车门)", "Car Mirror (汽车后视镜)", "License Plate (车牌)", "Steering Wheel (方向盘)",
    
    # -----------------------------------------------------------
    # 14. 🏥 医疗与健康 (Medical & Health) - [~30 items]
    # -----------------------------------------------------------
    "Stethoscope (听诊器)", "Thermometer (体温计)", "Blood Pressure Monitor (血压计)", "First Aid Kit (急救箱)", "Pill Bottle (药瓶)",
    "Bandage (创可贴)", "Syringe (注射器)", "Mask (口罩)", "Crutch (拐杖)",
    "Wheelchair (轮椅)", "Inhaler (吸入器)", "Oxygen Tank (氧气瓶)", "Eye Drops (眼药水)", "Toothbrush (牙刷)",
    "Toothpaste (牙膏)", "Dental Floss (牙线)", "Soap Bar (肥皂)", "Shampoo Bottle (洗发水瓶)",
    "Lotion Bottle (润肤露瓶)", "Perfume Bottle (香水瓶)", "Deodorant (除臭剂)", "Razor (剃须刀)", "Toilet Paper Roll (卷纸)",
    "Tissue Box (纸巾盒)", "Hand Sanitizer (洗手液)", "Rubbing Alcohol (医用酒精)", "Cotton Ball (棉球)", "Cotton Swab (棉签)",

    # -----------------------------------------------------------
    # 15. 🐶 宠物与动物 (Pets & Animals) - [~30 items]
    # -----------------------------------------------------------
    "Dog (狗)", "Cat (猫)", "Fish (鱼)", "Bird (鸟)", "Rabbit (兔子)",
    "Hamster (仓鼠)", "Turtle (乌龟)", "Snake (蛇)", "Lizard (蜥蜴)", "Spider (蜘蛛)",
    "Butterfly (蝴蝶)", "Ant (蚂蚁)", "Bee (蜜蜂)", "Ladybug (瓢虫)", "Dragonfly (蜻蜓)",
    "Dog Food Bowl (狗粮碗)", "Fish Tank (鱼缸)", "Bird Cage (鸟笼)", "Hamster Wheel (仓鼠轮)",

    # -----------------------------------------------------------
    # 16. 🏢 建筑与结构 (Buildings & Structures) - [~30 items]
    # -----------------------------------------------------------
    "House (房子)", "Apartment Building (公寓楼)", "Office Building (办公楼)", "School (学校)", "Hospital (医院)",
    "Church (教堂)", "Bridge (桥梁)", "Tunnel (隧道)", "Castle (城堡)",
    "Door (门)", "Window (窗户)", "Roof (屋顶)", "Chimney (烟囱)", "Fence (围栏)",
    "Wall (墙)", "Floor (地板)", "Ceiling (天花板)", "Staircase (楼梯)", "Elevator (电梯)",
    "Escalator (自动扶梯)",

    # -----------------------------------------------------------
    # 17. 🎨 艺术与工艺品 (Art & Crafts) - [~30 items]
    # -----------------------------------------------------------
    "Oil Painting (油画)", "Watercolor Painting (水彩画)", "Sculpture (雕塑)", "Pottery (陶器)",
    "Wood Carving (木雕)", "Paintbrush (画笔)", "Palette (调色板)",
    "Clay (黏土)", "Scissors (手工剪刀)", "Glue (胶水)", "Glitter (闪粉)",
    "Origami (折纸)", "Beads (珠子)", "Yarn (毛线)",  "Fabric (布料)", "Sewing Machine (缝纫机)", "Thread (线)",
    "Needle (针)", "Button (纽扣)", "Zipper (拉链)", "Ribbon (丝带)",

    # -----------------------------------------------------------
    # 18. 🌱 园艺与植物 (Gardening & Plants) - [~30 items]
    # -----------------------------------------------------------
    "Rose Bush (玫瑰灌木)", "Tomato Plant (番茄植株)", "Herb Garden (香草园)", "Flower Pot (花盆)",
    "Pruning Shears (修枝剪)", "Garden Trowel (园艺铲)", "Wheelbarrow (手推车)",
    "Watering Can (浇水壶)", "Garden Hose (花园水管)", "Plant Pot (植物盆)", "Plant Stand (植物架)", "Garden Bench (花园长椅)",
    "Flower Bed (花坛)", "Vegetable Garden (蔬菜园)", "Fruit Tree (果树)", "Shrub (灌木)", "Tree (树)",
    "Grass (草)", "Weed (杂草)", "Leaf (叶子)", "Flower (花)", "Seed (种子)",

    # -----------------------------------------------------------
    # 19. 🏋️ 运动与健身 (Exercise & Fitness) - [~30 items]
    # -----------------------------------------------------------
    "Yoga Mat (瑜伽垫)", "Dumbbell (哑铃)", "Barbell (杠铃)", "Resistance Band (阻力带)", "Treadmill (跑步机)",
    "Elliptical Machine (椭圆机)", "Stationary Bike (动感单车)", 
    "Water Bottle (运动水壶)", "Gym Bag (健身包)", "Sweat Towel (运动毛巾)", "Workout Clothes (运动服)", "Athletic Shoes (运动鞋)",
    "Jump Rope (跳绳)", "Pull-up Bar (引体向上杆)",
    "Boxing Gloves (拳击手套)", "Punching Bag (沙袋)",

    # -----------------------------------------------------------
    # 20. 🎄 节日与庆典 (Holidays & Celebrations) - [~30 items]
    # -----------------------------------------------------------
    "Christmas Tree (圣诞树)", "Halloween Pumpkin (万圣节南瓜)", "Birthday Cake (生日蛋糕)", "New Year's Fireworks (新年烟花)", "Easter Egg (复活节彩蛋)",
    "Party Hat (派对帽)", "Balloon (气球)", "Birthday Candle (生日蜡烛)", "Birthday Banner (生日横幅)",
    "Christmas Ornament (圣诞装饰)", "Christmas Lights (圣诞灯)", "Mistletoe (槲寄生)",
    "Easter Basket (复活节篮子)", "Valentine's Day Card (情人节卡片)",
    "Party Popper (派对拉炮)", "Streamer (彩带)", "Wedding Dress (婚纱)",

    # -----------------------------------------------------------
    # 21. 🔬 科学与实验 (Science & Experiments) - [~30 items]
    # -----------------------------------------------------------
    "Microscope (显微镜)", "Telescope (望远镜)", "Beaker (烧杯)", "Test Tube (试管)", "Petri Dish (培养皿)", "Magnifying Glass (放大镜)",
    "Compass (指南针)", "Ruler (尺子)", "Calculator (计算器)",
    "Thermometer (温度计)", "Barometer (气压计)", "Hygrometer (湿度计)",
    "Safety Goggles (护目镜)", "Lab Coat (实验服)", "Rubber Gloves (橡胶手套)", "Funnel (漏斗)",
    "Graduated Cylinder (量筒)", "Erlenmeyer Flask (锥形瓶)",

    # -----------------------------------------------------------
    # 22. 🛡️ 军事与防护 (Military & Protection) - [~30 items]
    # -----------------------------------------------------------
    "Helmet (头盔)", "Bulletproof Vest (防弹衣)", "Rifle (步枪)", "Hand Grenade (手榴弹)", "Tank (坦克)",
    "Jet Fighter (战斗机)", "Missile (导弹)", "Submarine (潜艇)", "Gas Mask (防毒面具)", "Shield (盾牌)",
    "Sword (剑)", "Armor (盔甲)", "Bow (弓)", "Arrow (箭)", "Crossbow (弩)",
    "Knife (刀)", "Bayonet (刺刀)", "Handgun (手枪)", "Shotgun (霰弹枪)", "Machine Gun (机关枪)",
    "Landmine (地雷)", "Cannon (大炮)", "Battleship (战舰)", "Aircraft Carrier (航空母舰)", "Missile Silo (导弹发射井)",
    "Camouflage Uniform (迷彩服)", "Military Boot (军靴)", "Night Vision Goggles (夜视镜)", "Binoculars (双筒望远镜)", "Walkie Talkie (对讲机)"
]